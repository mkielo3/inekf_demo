from copy import deepcopy
from scipy.linalg import block_diag
import numpy as np

from iekf_tests import run_tests
from iekf_logging import Logger


def so3_wedge(w):
	""" map vector ℝ³ to Lie algebra so(3) (via skew-symmetric matrix)"""
	return np.array([
		[0, -w[2], w[1]],
		[w[2], 0, -w[0]],
		[-w[1], w[0], 0]
	])


def so3_Exp(xi, _small_angle_tol = 1e-10):
	""" map from Lie algebra so(3) to Lie group SO(3) """
	phi = np.array(xi).ravel()
	angle = np.sqrt(phi.dot(phi))

	# For very small anglesu use Taylor series expansion
	if angle < _small_angle_tol:
		t2 = angle**2
		A = 1.0 - t2 / 6.0 * (1.0 - t2 / 20.0 * (1.0 - t2 / 42.0))
		B = (
			1.0
			/ 2.0
			* (1.0 - t2 / 12.0 * (1.0 - t2 / 30.0 * (1.0 - t2 / 56.0)))
		)
	else:
		A = np.sin(angle) / angle
		B = (1.0 - np.cos(angle)) / (angle**2)

	# Rodirgues rotation formula (103)
	Xi = so3_wedge(phi.ravel())
	return np.eye(3) + A * Xi + B * Xi.dot(Xi)


def sek3_Exp(v, small_angle_tol=1e-10):
	""" map from Lie algebra se_k(3) to Lie group SE_K(3)"""
	K = int((v.shape[0] - 3) / 3)  # Number of landmarks
	X = np.eye(3 + K)
	
	# Extract and compute rotation part using existing so3_Exp
	w = v[:3]
	R = so3_Exp(w, small_angle_tol)
	X[:3, :3] = R

	# Compute left Jacobian (could also be split into separate function)
	theta = np.linalg.norm(w)
	if theta < small_angle_tol:
		Jl = np.eye(3)
	else:
		A = so3_wedge(w)
		theta2 = theta * theta
		oneMinusCosTheta2 = (1 - np.cos(theta)) / theta2
		Jl = np.eye(3) + oneMinusCosTheta2 * A + ((theta - np.sin(theta)) / (theta2 * theta)) * A @ A

	# Transform landmarks
	for i in range(K):
		v_segment = v[3 + 3*i : 3 + 3*i + 3]
		X[:3, 3 + i] = Jl @ v_segment

	return X


def sek3_adjoint(X):
	""" compute adjoint map for SE_K(3) """
	K = X.shape[1] - 3
	R = X[0:3, 0:3]
	Adj = np.kron(np.eye(K + 1), R) # Initialize block diagonal of R matrices
	for i in range(K):
		Adj[3+3*i:3+3*i+3, 0:3] = so3_wedge(X[0:3, 3+i]) @ R
		
	return Adj


class Observation:

	def __init__(self, Y, b, H, N, PI):
		self.Y = np.concatenate(Y)
		self.b = np.concatenate(b)
		self.H = np.concatenate(H)
		self.N = N
		self.PI = PI


class RobotState:
	"""Maintains robot state & wraps interactions with X for convenience"""

	def __init__(self, X=np.eye(5), theta=np.zeros(6), P=None):
		self.X = X
		self.Theta_ = theta
		self.P_ = np.eye(3*X.shape[0]) if P is None else P

	def get_rotation(self): return self.X[0:3, 0:3]
	def get_velocity(self): return self.X[0:3, 3]
	def get_position(self): return self.X[0:3, 4]
	def get_gyro_bias(self): return self.Theta_[0:3]
	def get_accel_bias(self): return self.Theta_[-3:]
	def get_state(self): return self.X[0:3, 0:3], self.X[0:3, 3], self.X[0:3, 4]

	def set_rotation(self, R): self.X[0:3, 0:3] = R
	def set_velocity(self, v): self.X[0:3, 3] = v
	def set_position(self, p): self.X[0:3, 4] = p
	def set_gyro_bias(self, x): self.Theta_[0:3] = x
	def set_accel_bias(self, x): self.Theta_[-3:] = x
	def set_state(self, R, v, p): self.set_rotation(R); self.set_velocity(v); self.set_position(p)


class LandmarkManager:
	def __init__(self, prior_landmarks, Ql):
		self.prior_landmarks = prior_landmarks
		self.estimated_landmarks = {}
		self.Ql = Ql

	def process_measurements(self, measurements, current_state):
		"""
		Takes raw measurements and current state
		Returns: 
		- Observation object ready for IEKF correct() if measurements exist
		- List of new landmarks to add
		- None, None if no valid measurements
		"""
		R = current_state.get_rotation()
		Y, b, H, N, PI = [], [], [], [], np.array([]).reshape(0,0)
		new_landmarks = []
		used_landmarks = {}

		for (ld_id, ld_pos) in measurements:
			if ld_id in used_landmarks:
				continue
				
			used_landmarks[ld_id] = 1
			prior_pos = self.prior_landmarks.get(ld_id)
			estimated_idx = estimated_landmarks.get(ld_id)

			if prior_pos is not None:
				dim_X = current_state.X.shape[0]
				dim_P = current_state.P_.shape[0]

				# update Y
				Y_new = np.zeros(dim_X)
				Y_new[:3] = ld_pos
				Y_new[4] = 1
				Y.append(Y_new)

				# update b
				b_new = np.zeros(dim_X)
				b_new[:3] = prior_pos
				b_new[4] = 1
				b.append(b_new)

				# update H
				H_new = np.zeros((3, dim_P))
				H_new[:3, :3] = so3_wedge(prior_pos.ravel())
				H_new[:3, 6:9] = -np.eye(3)
				H.append(H_new)

				# update N
				N_new = R @ Ql @ R.T
				N = block_diag(N, N_new) if len(N) > 0 else N_new

				# Update PI
				PI_new = np.zeros((3, dim_X))
				PI_new[:3, :3] = np.eye(3)
				PI = block_diag(PI, PI_new) if len(PI) > 0 else PI_new

			elif estimated_idx:
				dim_X = current_state.X.shape[0]
				dim_P = current_state.P_.shape[0]

				# update Y
				Y_new = np.zeros(dim_X)
				Y_new[:3] = ld_pos
				Y_new[4] = 1
				Y_new[estimated_idx] = -1
				Y.append(Y_new)

				# update b
				b_new = np.zeros(dim_X)
				b_new[4] = 1
				b_new[estimated_idx] = -1
				b.append(b_new)

				# update H
				H_new = np.zeros((3, dim_P))
				H_new[:3, 6:9] = -np.eye(3)
				H_new[:3, 3*estimated_idx-6:3*estimated_idx-3] = np.eye(3)
				H.append(H_new)

				# update N
				N_new = R @ Ql @ R.T
				N = block_diag(N, N_new) if len(N) > 0 else N_new

				# Update PI
				PI_new = np.zeros((3, dim_X))
				PI_new[:3, :3] = np.eye(3)
				PI = block_diag(PI, PI_new) if len(PI) > 0 else PI_new

			else:
				new_landmarks.append((ld_id, ld_pos))

		if len(Y) > 0:
			return Observation(Y, b, H, N, PI), new_landmarks

		return None, None

	def add_new_landmarks(self, landmarks, current_state):
		"""
		Handles state augmentation for new landmarks
		Returns updated state and covariance
		"""
		X_aug = current_state.X
		P_aug = current_state.P_
		p = current_state.get_position()

		for (ld_id, ld_pos) in landmarks:
			dim_P = current_state.P_.shape[0]
			dim_theta = current_state.Theta_.shape[0]

			# Initialize new landmark mean
			startIndex = X_aug.shape[0]
			X_aug = np.pad(X_aug, ((0,1), (0,1)))
			X_aug[startIndex, startIndex] = 1
			X_aug[:3, startIndex] = p + R @ ld_pos

			# Initialize new landmark covariance
			F = np.zeros((dim_P + 3, dim_P))
			F[:dim_P-dim_theta, :dim_P-dim_theta] = np.eye(dim_P-dim_theta)
			F[dim_P-dim_theta:dim_P-dim_theta+3, 6:9] = np.eye(3)
			F[dim_P-dim_theta+3:, dim_P-dim_theta:] = np.eye(dim_theta)

			G = np.zeros((F.shape[0], 3))
			G[-dim_theta-3:-dim_theta, :] = R

			P_aug = F @ P_aug @ F.T + G @ Ql @ G.T

			# Add to estimated landmarks
			estimated_landmarks[ld_id] = startIndex

			# Update state and covariance
		
		return X_aug, P_aug


# Create all covariance matrices, with noise values to match C++
gyro_std = 0.01  # Gyroscope noise
accel_std = 0.1  # Accelerometer noise 
landmark_std = 0.1  # Landmark noise

def make_cov(std):
	return std*std * np.eye(3)

Qg = make_cov(gyro_std)  # Gyroscope covariance
Qa = make_cov(accel_std)  # Accelerometer covariance 
Ql = make_cov(landmark_std)  # Landmark covariance


# Initialize state and filter (same as before)
state = RobotState()
state.set_state(R=np.array([[1,0,0], [0,-1,0], [0,0,-1]]), 
				v=np.array([0,0,0]), 
				p=np.array([0,0,0]))

state.set_gyro_bias(np.array([0,0,0]))
state.set_accel_bias(np.array([0,0,0]))

# Initialize environments: 1) gravity, 2) known landmarks (leave landmark 2 as unknown) 3) dictionary to hold unknown landmarks
g_ = np.array([0, 0, -9.81]) # gravity
lm1 = (1, np.array([0, -1, 0]))
lm3 = (3, np.array([2, -1, 0.5]))
known_landmarks = {lm1[0]: lm1[1], lm3[0]: lm3[1]}  # known landmarks
estimated_landmarks = {} # estimated landmarks
landmark_manager = LandmarkManager(known_landmarks, Ql)


# Set rate to process data
dt_min, dt_max = 1e-6, 1
t = 0
t_prev = 0
imu_measurement_prev = np.zeros(6)
imu_measurement = np.zeros(6)

# Read and process measurements
logger = Logger()
input_file = "data/input/imu_landmark_measurements.txt"
with open(input_file, 'r') as f:
	for line in f.readlines():
		splits = line.split(" ")
		if splits[0] == 'IMU':

			t, imu_measurement = float(splits[1]), np.array(splits[2:8], dtype='float')
			dt = t - t_prev

			if (dt > dt_min and dt < dt_max):

				###############################
				# Start of Propogate Function #
				###############################

				X, P_ = deepcopy(state.X), deepcopy(state.P_)

				# Adjust IMU measurements by bias
				w = imu_measurement_prev[:3] - state.get_gyro_bias()
				a = imu_measurement_prev[-3:] - state.get_accel_bias()

				# Extract state
				R, v, p = state.get_state()

				# Apply strapdown IMU motion model
				phi = w * dt
				R_pred = R @ so3_Exp(phi)
				v_pred = v + (R @ a + g_)*dt
				p_pred = p + v * dt + 0.5 * (R @ a + g_) * dt * dt

				# Set new state
				state.set_state(R=R_pred, v=v_pred, p=p_pred)

				# Linearize error dynamics
				dim_X, dim_P, dim_theta = X.shape[0], P_.shape[0], state.Theta_.shape[0]

				# Inertial terms
				A = np.zeros_like(P_)
				A[3:6, 0:3] = so3_wedge(g_.ravel())
				A[6:9, 3:6] = np.eye(3)

				# Bias
				A[0:3, dim_P-dim_theta:dim_P-dim_theta+3] = -R
				A[3:6, dim_P-dim_theta+3:dim_P-dim_theta+6] = -R
				for i in range(3, dim_X):
					A[3*i-6:3*i-3, dim_P-dim_theta:dim_P-dim_theta+3] = -so3_wedge(X[0:3, i].ravel()) @ R

				# Noise terms
				Qk = np.zeros_like(P_)
				Qk[0:3, 0:3] = Qg  # Gyroscope block
				Qk[3:6, 3:6] = Qa  # Accelerometer block

				I = np.eye(dim_P)
				Phi = I + A * dt

				Adj = np.eye(dim_P)  # Identity matrix of size dim_P
				Adj[:dim_P-dim_theta, :dim_P-dim_theta] = sek3_adjoint(X)
				PhiAdj = Phi @ Adj
				Qk_hat = PhiAdj @ Qk @ PhiAdj.T * dt

				# Propagate Covariance
				P_pred = Phi @ P_ @ Phi.T + Qk_hat
				state.P_ = P_pred

				###############################
				#  End of Propogate Function  #
				###############################

				logger.log_flat(t, state, estimated_landmarks)

		elif splits[0] == 'LANDMARK':
			t = float(splits[1])

			measured_landmarks = []
			for i in range(2, len(splits), 4):
				idx = float(splits[i])
				coords = np.array(splits[i+1:i+4], dtype='float')
				measured_landmarks.append((idx, coords))

			# Get observation from landmark manager
			obs, new_landmarks = landmark_manager.process_measurements(measured_landmarks, state)

			if obs is not None:

				##################################
				#~ Start of Correction Function ~#
				##################################

				P = state.P_
				PHT = P @ obs.H.T

				S = obs.H @ PHT + obs.N
				K = PHT @ np.linalg.inv(S)

				n_measurements = obs.Y.shape[0] // state.X.shape[0]
				BigX = np.kron(np.eye(n_measurements), state.X)
				
				# Compute correction terms
				Z = BigX @ obs.Y - obs.b
				delta = K @ obs.PI @ Z
				dim_theta = state.Theta_.shape[0]
				dX = sek3_Exp(delta[:-dim_theta])
				dTheta = delta[-dim_theta:]

				# Update state
				X_new = dX @ state.X  # Right-Invariant Update
				Theta_new = state.Theta_ + dTheta
				state.X = X_new
				state.Theta_ = Theta_new

				# Update Covariance (Joseph form)
				IKH = np.eye(state.P_.shape[0]) - K @ obs.H
				P_new = IKH @ P @ IKH.T + K @ obs.N @ K.T

				state.P_ = P_new

				##################################
				#~  End of Correction Function  ~#
				##################################

			# Handle new landmarks if any
			if new_landmarks:
				state.X, state.P_ = landmark_manager.add_new_landmarks(new_landmarks, state)
			
			logger.log_flat(t, state, estimated_landmarks)

		t_prev = t
		imu_measurement_prev = imu_measurement

logger.save()
run_tests() 
