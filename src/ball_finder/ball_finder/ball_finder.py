import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np

class YellowSphereDetector(Node):
    def __init__(self):
        super().__init__('yellow_sphere_detector')
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',  # Update this topic to match your camera's image topic
            self.image_callback,
            10
        )

        # Publisher to the TurtleBot3's velocity command topic
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Initialize a timer to check for sphere detection and control robot
        self.timer = self.create_timer(0.1, self.control_loop)

        # Store the centroid of the detected sphere
        self.sphere_center = None

        # Camera frame dimensions (set later)
        self.frame_width = None
        self.frame_height = None

        self.done = False

    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        
        # Set frame dimensions
        if self.frame_width is None or self.frame_height is None:
            self.frame_height, self.frame_width, _ = frame.shape
        
        # Convert the image to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define the range for yellow color in HSV
        lower_yellow = np.array([10, 70, 70])
        upper_yellow = np.array([30, 255, 255])
        
        # Threshold the HSV image to get only yellow colors
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            if area > 500:  # Minimum area to consider a valid sphere
                # Calculate the centroid of the largest contour
                M = cv2.moments(largest_contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    self.sphere_center = (cx, cy)
                    return
        
        # If no valid sphere is detected, set sphere_center to None
        self.sphere_center = None
    
    def control_loop(self):
        twist = Twist()

        if self.sphere_center:
            cx, _ = self.sphere_center
            
            # Calculate error in the x direction (difference from the center of the frame)
            error_x = cx - (self.frame_width / 2)
            
            # Rotation direction based on error
            if abs(error_x) > 20:  # Allowable threshold
                twist.angular.z = -0.002 * error_x  # Rotate proportionally to the error
                twist.linear.x = 0.1  # Move forward slowly
            
            # Move forward if the sphere is near the center
            if abs(error_x) <= 20:
                twist.linear.x = 0.2  # Move forward slowly
        else:
            # No sphere detected, rotate right
            twist.angular.z = -0.3

        # Publish the velocity command
        self.cmd_vel_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = YellowSphereDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
