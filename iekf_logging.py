import pandas as pd


class Logger:
   def __init__(self):
       self.log_data = []

   def log_filter(self, t, filter):
       pos = filter.state.get_position()
       vel = filter.state.get_velocity() 
       rot = filter.state.get_rotation()
       
       self.log_data.append({
           'timestamp': t,
           'object': 'robot',
           'pos_x': pos[0],
           'pos_y': pos[1], 
           'pos_z': pos[2],
           'vel_x': vel[0],
           'vel_y': vel[1],
           'vel_z': vel[2],
           'rot_11': rot[0,0],
           'rot_12': rot[0,1],
           'rot_13': rot[0,2],
           'rot_21': rot[1,0],
           'rot_22': rot[1,1],
           'rot_23': rot[1,2],
           'rot_31': rot[2,0],
           'rot_32': rot[2,1],
           'rot_33': rot[2,2]
       })

       for ld_id, col_idx in filter.estimated_landmarks.items():
           landmark_pos = filter.state.X[:3, col_idx]
           self.log_data.append({
               'timestamp': t,
               'object': f'landmark_{int(ld_id)}',
               'pos_x': landmark_pos[0],
               'pos_y': landmark_pos[1],
               'pos_z': landmark_pos[2],
               'vel_x': None,
               'vel_y': None,
               'vel_z': None,
               'rot_11': None,
               'rot_12': None,
               'rot_13': None,
               'rot_21': None,
               'rot_22': None,
               'rot_23': None,
               'rot_31': None,
               'rot_32': None,
               'rot_33': None
           })

   def log_flat(self, t, state, estimated_landmarks):
       pos = state.get_position()
       vel = state.get_velocity() 
       rot = state.get_rotation()
       
       self.log_data.append({
           'timestamp': t,
           'object': 'robot',
           'pos_x': pos[0],
           'pos_y': pos[1], 
           'pos_z': pos[2],
           'vel_x': vel[0],
           'vel_y': vel[1],
           'vel_z': vel[2],
           'rot_11': rot[0,0],
           'rot_12': rot[0,1],
           'rot_13': rot[0,2],
           'rot_21': rot[1,0],
           'rot_22': rot[1,1],
           'rot_23': rot[1,2],
           'rot_31': rot[2,0],
           'rot_32': rot[2,1],
           'rot_33': rot[2,2]
       })

       for ld_id, col_idx in estimated_landmarks.items():
           landmark_pos = state.X[:3, col_idx]
           self.log_data.append({
               'timestamp': t,
               'object': f'landmark_{int(ld_id)}',
               'pos_x': landmark_pos[0],
               'pos_y': landmark_pos[1],
               'pos_z': landmark_pos[2],
               'vel_x': None,
               'vel_y': None,
               'vel_z': None,
               'rot_11': None,
               'rot_12': None,
               'rot_13': None,
               'rot_21': None,
               'rot_22': None,
               'rot_23': None,
               'rot_31': None,
               'rot_32': None,
               'rot_33': None
           })

   def save(self, filename='data/output/output.csv'):
       df = pd.DataFrame(self.log_data)
       df.to_csv(filename, float_format='%.15f', index=False)