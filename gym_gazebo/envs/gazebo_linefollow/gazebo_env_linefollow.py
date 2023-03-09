
import cv2
import gym
import math
import rospy
import roslaunch
import time
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import Image
from time import sleep

from gym.utils import seeding


class Gazebo_Linefollow_Env(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        LAUNCH_FILE = '/home/fizzer/enph353_gym-gazebo-noetic/gym_gazebo/envs/ros_ws/src/linefollow_ros/launch/linefollow_world.launch'
        gazebo_env.GazeboEnv.__init__(self, LAUNCH_FILE)
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world',
                                              Empty)

        self.action_space = spaces.Discrete(3)  # F,L,R
        self.reward_range = (-np.inf, np.inf)
        self.episode_history = []

        self._seed()

        self.bridge = CvBridge()
        self.timeout = 0  # Used to keep track of images with no line detected


    def process_image(self, data):
        '''
            @brief Coverts data into a opencv image and displays it
            @param data : Image data from ROS

            @retval (state, done)
        '''
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # cv2.imshow("raw", cv_image)

        NUM_BINS = 3
        state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        done = False

        # TODO: Analyze the cv_image and compute the state array and
        # episode termination condition.
        #
        # The state array is a list of 10 elements indicating where in the
        # image the line is:
        # i.e.
        #    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] indicates line is on the left
        #    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] indicates line is in the center
        #
        # The episode termination condition should be triggered when the line
        # is not detected for more than 30 frames. In this case set the done
        # variable to True.
        #
        # You can use the self.timeout variable to keep track of which frames
        # have no line detected.
        
        # ORIGINAL IDEA:
        # grayScale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # radius = 20

        # ret, mask = cv2.threshold(grayScale,70,255,cv2.THRESH_BINARY_INV)
        # strip = mask[-radius,:,]

        # road = []

        # for i in range(len(strip)):
        #     if(strip[i] == 255):
        #         road.append(i)

        # if len(road) == 0:
        #     center_x = None
        # else:
        #     road = np.array(road)
        #     center_x = int(np.mean(road))
        
        # h, w, c = frame.shape
        # center_y = h - radius

        # image = np.copy(frame)

        # cv2.circle(img=image, center = (center_x,center_y), radius = radius, color =(0,0,255), thickness=-1)

        # cv2_imshow(image)
        # out.write(image)

        def detect_line(frame):
            cv_image = frame
            ## get frame and binary/mask
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            threshold = 200
            _, img_bin = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY)

            ## line detection algorithm
            ## essentially starting near the bottom and finding the middle black pixel 
            ##  value for a row, then using those coordinates to draw a dot
            min = -1
            ycoord = 0

            cv2.imshow("im", gray)
            cv2.waitKey(1)
            dots = []

            for y in range(10):
                if min != -1:
                    ycoord = y
                    break
                for x in range(shape[1]):
                    if (gray[shape[0]-y-1][x] < threshold):
                        dots.append(x)
                        if len(dots) >= 2:
                            min = x
            if len(dots) > 1:
                avg = math.floor(np.median(dots))
            else:
                avg = 0
            
            # cv2.circle(cv_image, (avg, shape[0]-ycoord-10), 10, (0,0,0), 3)
            # cv2.imshow("raw", cv_image)
            # cv2.waitKey(1)

            return avg

        shape = np.shape(cv_image)
        division = math.floor(shape[1]/10)

        # partitions = []
        # for i in range(10):
        #     partitions.append(cv_image[math.floor(shape[0]*2/3):math.floor(shape[0]), i*division:(i+1)*division])
        # weight = []
        # for j in range(10):
        #     partition_weight = 0
        #     grayscale = cv2.cvtColor(partitions[j], cv2.COLOR_BGR2GRAY)
        #     for k in range(len(grayscale)):
        #         for l in range(len(grayscale[k])):
        #             if grayscale[k][l] >= THRESHOLD:
        #                 partition_weight += 1
        #     weight.append(partition_weight)

        output = detect_line(cv_image)

        if output == 0:
            self.timeout += 1
        else: 
            self.timeout = 0
        if self.timeout >= 30:
            done = True

        state[int(math.floor(output/division))] = 1
    
        return state, done

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        self.episode_history.append(action)

        vel_cmd = Twist()

        if action == 0:  # FORWARD
            vel_cmd.linear.x = 0.4
            vel_cmd.angular.z = 0.0
        elif action == 1:  # LEFT
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = 0.5
        elif action == 2:  # RIGHT
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = -0.5

        self.vel_pub.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/pi_camera/image_raw', Image,
                                              timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state, done = self.process_image(data)

        # Set the rewards for your action
        if not done:
            if action == 0:  # FORWARD
                reward = 4
            elif action == 1:  # LEFT
                reward = 2
            else:
                reward = 2  # RIGHT
        else:
            reward = -200

        return state, reward, done, {}

    def reset(self):

        print("Episode history: {}".format(self.episode_history))
        self.episode_history = []
        print("Resetting simulation...")
        # Resets the state of the environment and returns an initial
        # observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            # reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            # resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        # read image data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/pi_camera/image_raw',
                                              Image, timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        self.timeout = 0
        state, done = self.process_image(data)

        return state