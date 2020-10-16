"""
VRLog classes that write/read iGibson VR data to/from HDF5.

TODO: Save velocity/torque for algorithmic training? Not necessary for replay, but might be helpful.
Can easily save velocity for joints, but might have to use link states for normal pybullet objects.

HDF5 hierarchy:
/ (root)

--- action (group)

------ N x action_path (group) - these are paths introduced by the user and are of the form Y x group_name + dataset name

--- frame_data (group)

------ frame_number (dataset)
--------- DATA: int

------ last_frame_time (dataset) - the time the last frame took to simulate and render
--------- DATA: float

--- physics_data (group)

------ body_id_n (dataset, n is positive integer)
--------- DATA: [pos, orn, joint values] (len 7 + M, where M is number of joints)

--- vr (group)

------ vr_camera (group)

Note: we only need one eye to render in VR - we choose the right eye, since that is what is used to create
the computer's display when the VR is running
--------- right_eye_view (dataset)
------------ DATA: 4x4 mat
--------- right_eye_proj (dataset)
------------ DATA: 4x4 mat

------ vr_device_data (group)

--------- hmd (dataset)
------------ DATA: [is_valid, trans, rot] (len 8)
--------- left_controller (dataset)
------------ DATA: [is_valid, trans, rot] (len 8)
--------- right_controller (dataset)
------------ DATA: [is_valid, trans, rot] (len 8)

------ vr_button_data (group)

--------- left_controller (dataset)
------------ DATA: [trig_frac, touch_x, touch_y] (len 3)
--------- right_controller (dataset)
------------ DATA: [trig_frac, touch_x, touch_y] (len 3)

------ vr_eye_tracking_data (dataset)
--------- DATA: [is_valid, origin, dir, left_pupil_diameter, right_pupil_diameter] (len 9)
"""

import h5py
import numpy as np
import pybullet as p
import time

class VRLogWriter():
    """Class that handles saving of VR data, physics data and user-defined actions.

    Function flow:
    1) Before simulation
    init -> N x register_action -> set_up_data_storage

    2) During simulation:
    N x save_action (at any point during frame) -> process_frame (at end of frame)

    3) After simulation, before disconnecting from PyBullet sever:
    end_log_session
    """

    # TIMELINE: Initialize the VRLogger just before simulation starts, once all bodies have been loaded
    def __init__(self, frames_before_write, log_filepath, profiling_mode=False):
        # The number of frames to store data on the stack before writing to HDF5.
        # We buffer and flush data like this to cause a small an impact as possible
        # on the VR frame-rate.
        self.frames_before_write = frames_before_write
        # File path to write log to (path relative to script location)
        self.log_filepath = log_filepath
        # If true, will print out time it takes to save to hd5
        self.profiling_mode = profiling_mode
        # PyBullet body ids to be saved
        # TODO: Make sure this is the correct way to get the body ids!
        self.pb_ids = [p.getBodyUniqueId(i) for i in range(p.getNumBodies())]
        self.pb_id_data_len_map = dict()
        self.data_map = None
        # Sentinel that indicates a certain value was not set in the HDF5 
        self.default_fill_sentinel = -1.0
        # Counts number of frames (reset to 0 every self.frames_before_write)
        self.frame_counter = 0
        # Counts number of frames and does not reset
        self.persistent_frame_count = 0
        # Time when last frame ended (not valid for first frame, so set to 0)
        self.last_frame_end_time = 0
        # Handle of HDF5 file
        self.hf = None
        # Name path data - used to extract data from data map and save to hd5
        self.name_path_data = []
        self.generate_name_path_data()
        # Create data map
        self.create_data_map()

    def generate_name_path_data(self):
        """Generates lists of name paths for resolution in hd5 saving.
        Eg. ['vr', 'vr_camera', 'right_eye_view']."""
        self.name_path_data.extend([
                ['frame_data', 'frame_number'],
                ['frame_data', 'last_frame_duration'],
        ])

        for n in self.pb_ids:
            self.name_path_data.append(['physics_data', 'body_id_{0}'.format(n)])
        
        self.name_path_data.extend([
                ['vr', 'vr_camera', 'right_eye_view'],
                ['vr', 'vr_camera', 'right_eye_proj'],
                ['vr', 'vr_device_data', 'hmd'],
                ['vr', 'vr_device_data', 'left_controller'],
                ['vr', 'vr_device_data', 'right_controller'],
                ['vr', 'vr_button_data', 'left_controller'],
                ['vr', 'vr_button_data', 'right_controller'],
                ['vr', 'vr_eye_tracking_data'],
        ])

    def create_data_map(self):
        """Creates data map of data that will go into HDF5 file. All the data in the
        map is reset after every self.frames_before_write frames, by refresh_data_map."""
        self.data_map = dict()
        self.data_map['action'] = dict()
        self.data_map['frame_data'] = dict()
        self.data_map['frame_data']['frame_number'] = np.full((self.frames_before_write, 1), self.default_fill_sentinel)
        self.data_map['frame_data']['last_frame_duration'] = np.full((self.frames_before_write, 1), self.default_fill_sentinel)

        self.data_map['physics_data'] = dict()
        for pb_id in self.pb_ids:
            # pos + orn + number of joints
            array_len = 7 + p.getNumJoints(pb_id)
            self.pb_id_data_len_map[pb_id] = array_len
            self.data_map['physics_data']['body_id_{0}'.format(pb_id)] = np.full((self.frames_before_write, array_len), self.default_fill_sentinel)

        self.data_map['vr'] = {
            'vr_camera': {
                'right_eye_view': np.full((self.frames_before_write, 4, 4), self.default_fill_sentinel),
                'right_eye_proj': np.full((self.frames_before_write, 4, 4), self.default_fill_sentinel)
            }, 
            'vr_device_data': {
                'hmd': np.full((self.frames_before_write, 8), self.default_fill_sentinel),
                'left_controller': np.full((self.frames_before_write, 8), self.default_fill_sentinel),
                'right_controller': np.full((self.frames_before_write, 8), self.default_fill_sentinel)
            },
            'vr_button_data': {
                'left_controller': np.full((self.frames_before_write, 3), self.default_fill_sentinel),
                'right_controller': np.full((self.frames_before_write, 3), self.default_fill_sentinel)
            },
            'vr_eye_tracking_data': np.full((self.frames_before_write, 9), self.default_fill_sentinel)
        }
    
    # TIMELINE: Register all actions immediately after calling init
    def register_action(self, action_path, action_shape):
        """Registers an action to be saved every frame in the VRLogWriter.

        Args:
            action_path: The /-separated path specifying where to save action data. All entries but the last will be treated
                as group names, and the last entry will the be the dataset. The parent group for all
                actions is called action. Eg. action_path = vr_hand/constraint. This will end up in
                action (group) -> vr_hand (group) -> constraint (dataset) in the saved data.
            action_shape: tuple representing action shape. It is expected that all actions will be numpy arrays. They
                are stacked over time in the first dimension to create a persistent action data store.
        """
        # Extend name path data - this is used for fast saving and lookup later on
        act_path = ['action']
        path_tokens = action_path.split('/')
        act_path.extend(path_tokens)
        self.name_path_data.append(act_path)

        # Add action to dictionary - create any new dictionaries that don't yet exist
        curr_dict = self.data_map['action']
        for tok in path_tokens[:-1]:
            if tok in curr_dict.keys():
                curr_dict = curr_dict[tok]
            else:
                curr_dict[tok] = dict()
                curr_dict = curr_dict[tok]

        # Curr_dict refers to the last group - we then add in the dataset of the right shape
        # The action is extended across self.frames_before_write rows
        extended_shape = (self.frames_before_write,) + action_shape
        curr_dict[path_tokens[-1]] = np.full(extended_shape, self.default_fill_sentinel)

    # TIMELINE: Call set_up once all actions have been registered, or directly after init if no actions to save
    def set_up_data_storage(self):
        """Performs set up of internal data structures needed for storage, once
        VRLogWriter has been initialized and all actions have been registered."""
        # Note: this erases the file contents previously stored as self.log_filepath
        hf = h5py.File(self.log_filepath, 'w')
        for name_path in self.name_path_data:
            joined_path = '/'.join(name_path)
            curr_data_shape = (0,) + self.get_data_for_name_path(name_path).shape[1:]
            # None as first shape value allows dataset to grow without bound through time
            max_shape = (None,) + curr_data_shape[1:]
            # Create_dataset with a '/'-joined path automatically creates the required groups
            hf.create_dataset(joined_path, curr_data_shape, maxshape=max_shape)
        
        hf.close()
        # Now open in r+ mode to append to the file
        self.hf = h5py.File(self.log_filepath, 'r+')

    def get_data_for_name_path(self, name_path):
        """Resolves a list of names (group/dataset) into a numpy array.
        eg. [vr, vr_camera, right_eye_view] -> self.data_map['vr']['vr_camera']['right_eye_view']"""
        next_data = self.data_map
        for name in name_path:
            next_data = next_data[name]

        return next_data

    # TIMELINE: Call this at any time before process_frame to save a specific action
    def save_action(self, action_path, action):
        """Saves a single action to the VRLogWriter. It is assumed that this function will
        be called every frame, including the first.

        Args:
            action_path: The /-separated action path that was used to register this action
            action: The action as a numpy array - must have the same shape as the action_shape that
                was registered along with this action path
        """
        full_action_path = 'action/' + action_path
        act_data = self.get_data_for_name_path(full_action_path.split('/'))
        act_data[self.frame_counter, ...] = action

    def write_frame_data_to_map(self):
        """Writes frame data to the data map."""
        self.data_map['frame_data']['frame_number'][self.frame_counter, ...] = self.persistent_frame_count
        self.data_map['frame_data']['last_frame_duration'][self.frame_counter, ...] = time.time() - self.last_frame_end_time
        self.last_frame_end_time = time.time()

    def write_vr_data_to_map(self, s):
        """Writes all VR data to map. This will write data
        that the user has not even processed in their demos.
        For example, we will store eye tracking data if it is
        valid, even if they do not explicitly use that data
        in their code. This will help us store all the necessary
        data without remembering to call the simulator's data
        extraction functions every time we want to save data.

        Args:
            s (simulator): used to extract information about VR system
        """
        # At end of each frame, renderer has camera information for VR right eye
        self.data_map['vr']['vr_camera']['right_eye_view'][self.frame_counter, ...] = s.renderer.V
        self.data_map['vr']['vr_camera']['right_eye_proj'][self.frame_counter, ...] = s.renderer.P

        for device in ['hmd', 'left_controller', 'right_controller']:
            is_valid, trans, rot = s.getDataForVRDevice(device)
            if is_valid is not None:
                data_list = [is_valid]
                data_list.extend(trans)
                data_list.extend(rot)    
                self.data_map['vr']['vr_device_data'][device][self.frame_counter, ...] = np.array(data_list)

            if device == 'left_controller' or device == 'right_controller':
                button_data_list = s.getButtonDataForController(device)
                if button_data_list[0] is not None:
                    self.data_map['vr']['vr_button_data'][device][self.frame_counter, ...] = np.array(button_data_list)

        is_valid, origin, dir, left_pupil_diameter, right_pupil_diameter = s.getEyeTrackingData()
        if is_valid is not None:
            eye_data_list = [is_valid]
            eye_data_list.extend(origin)
            eye_data_list.extend(dir)
            eye_data_list.append(left_pupil_diameter)
            eye_data_list.append(right_pupil_diameter)
            self.data_map['vr']['vr_eye_tracking_data'][self.frame_counter, ...] = np.array(eye_data_list)

    def write_pybullet_data_to_map(self):
        """Write all pybullet data to the class' internal map."""
        for pb_id in self.pb_ids:
            data_list = []
            pos, orn = p.getBasePositionAndOrientation(pb_id)
            data_list.extend(pos)
            data_list.extend(orn)
            data_list.extend([p.getJointState(pb_id, n)[0] for n in range(p.getNumJoints(pb_id))])
            self.data_map['physics_data']['body_id_{0}'.format(pb_id)][self.frame_counter] = np.array(data_list)

    # TIMELINE: Call this at the end of each frame (eg. at end of while loop)
    def process_frame(self, s):
        """Asks the VRLogger to process frame data. This includes:
        -- updating pybullet data
        -- incrementing frame counter by 1

        Args:
            s (simulator): used to extract information about VR system
        """
        self.write_frame_data_to_map()
        self.write_vr_data_to_map(s)
        self.write_pybullet_data_to_map()
        self.frame_counter += 1
        self.persistent_frame_count += 1
        if (self.frame_counter >= self.frames_before_write):
            self.frame_counter = 0
            # We have accumulated enough data, which we will write to hd5
            self.write_to_hd5()

    def refresh_data_map(self):
        """Resets all values stored in self.data_map to the default sentinel value.
        This function is called after we have written the last self.frames_before_write
        frames to HDF5 and can start inputting new frame data into the data map."""
        for name_path in self.name_path_data:
            np_data = self.get_data_for_name_path(name_path)
            np_data.fill(self.default_fill_sentinel)

    def write_to_hd5(self):
        """Writes data stored in self.data_map to hd5.
        The data is saved each time this function is called, so data
        will be saved even if a Ctrl+C event interrupts the program."""
        print('----- Writing log data to hd5 on frame: {0} -----'.format(self.persistent_frame_count))
        start_time = time.time()
        for name_path in self.name_path_data:
            curr_dset = self.hf['/'.join(name_path)]
            # Resize to accommodate new data
            curr_dset.resize(curr_dset.shape[0] + self.frames_before_write, axis=0)
            # Set last self.frames_before_write rows to numpy data from data map
            curr_dset[-self.frames_before_write:, ...] = self.get_data_for_name_path(name_path)

        self.refresh_data_map()
        delta = time.time() - start_time
        if self.profiling_mode:
            print('Time to write: {0}'.format(delta))

    def end_log_session(self):
        """Closes hdf5 log file at end of logging session."""
        self.hf.close()

class VRLogReader():
    # TIMELINE: Initialize the VRLogReader before reading any frames
    def __init__(self, log_filepath):
        self.log_filepath = log_filepath
        # Frame counter keeping track of how many frames have been reproduced
        self.frame_counter = 0
        self.hf = h5py.File(self.log_filepath, 'r')
        self.pb_ids = self.extract_pb_ids()
        # Get total frame num (dataset row length) from an arbitary dataset
        self.total_frame_num = self.hf['vr/vr_device_data/hmd'].shape[0]
        # Boolean indicating if we still have data left to read
        self.data_left_to_read = True
        print('----- VRLogReader initialized -----')
        print('Preparing to read {0} frames'.format(self.total_frame_num))

    def extract_pb_ids(self):
        """Extracts pybullet body ids from saved data."""
        return [int(metadata[0].split('_')[-1]) for metadata in self.hf['physics_data'].items()]

    def read_frame(self, s, fullReplay=True):
        """Reads a frame from the VR logger and steps simulation with stored data."""

        """Reads a frame from the VR logger and steps simulation with stored data.
        This includes the following two steps:
        -- update camera data
        -- update pybullet physics data

        Args:
            s (simulator): used to set camera view and projection matrices
            fullReplay: boolean indicating if we should replay full state of the world or not.
                If this value is set to false, we simply increment the frame counter each frame, 
                and let the user take control of processing actions and simulating them
        """
        # Note: Currently returns hmd position, as a test
        # Catch error where the user tries to keep reading a frame when all frames have been read
        if self.frame_counter >= self.total_frame_num:
            return

        # Get recorded frame duration for this frame
        frame_duration = self.hf['frame_data']['last_frame_duration'][self.frame_counter]

        read_start_time = time.time()
        # Each frame we first set the camera data
        s.renderer.V = self.hf['vr/vr_camera/right_eye_view'][self.frame_counter]
        s.renderer.P = self.hf['vr/vr_camera/right_eye_proj'][self.frame_counter]

        if fullReplay:
            # If doing full replay we update the physics manually each frame
            for pb_id in self.pb_ids:
                id_name = 'body_id_{0}'.format(pb_id)
                id_data = self.hf['physics_data/' + id_name][self.frame_counter]
                pos = id_data[:3]
                orn = id_data[3:7]
                joint_data = id_data[7:]
                p.resetBasePositionAndOrientation(pb_id, pos, orn)
                for i in range(len(joint_data)):
                    p.resetJointState(pb_id, i, joint_data[i])

        self.frame_counter += 1
        if self.frame_counter >= self.total_frame_num:
            self.data_left_to_read = False
            self.end_log_session()
        
        if fullReplay:
            # Only sleep to simulate accurate timestep if doing full replay
            read_duration = time.time() - read_start_time
            # Sleep to match duration of this frame, to create an accurate replay
            if read_duration < frame_duration:
                time.sleep(frame_duration - read_duration)

    def read_action(self, action_path):
        """Reads the action at action_path for the current frame.

        Args:
            action_path: /-separated string representing the action to fetch. This should match
                an action that was previously registered with the VRLogWriter during data saving
        """
        full_action_path = 'action/' + action_path
        return self.hf[full_action_path][self.frame_counter]
    
    # TIMELINE: Use this as the while loop condition to keep reading frames!
    def get_data_left_to_read(self):
        """Returns whether there is still data left to read."""
        return self.data_left_to_read
    
    def end_log_session(self):
        """This is called once reading has finished to clean up resources used."""
        print('Ending frame reading session after reading {0} frames'.format(self.total_frame_num))
        self.hf.close()
        print('----- VRLogReader shutdown -----')