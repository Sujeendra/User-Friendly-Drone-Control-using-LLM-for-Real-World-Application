# -*- coding: utf-8 -*-
import sys
import rospy
from geometry_msgs.msg import PoseStamped, Quaternion
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QLabel
from PyQt5.QtCore import Qt
from nav_msgs.msg import Odometry
import math
from tf.transformations import euler_from_quaternion
from guidance import models, gen, system, user, assistant
import guidance
import os
import json

class llm:

    def __init__(self):
        os.environ['OPENAI_API_KEY'] = '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'
        self.end = None
        self.model = models.OpenAI("gpt-3.5-turbo")
        self.llm_chain = self.model

    def init_llm(self):
        with system():
            self.llm_chain += f"""\
                    You need to serve as a robot that takes input from the user which can contain shapes which a drone needs to follow. Your purpose is to give certain coordinates as output which will be implemented on the drone, the rubrics for the output coordinates will be mentioned below.

                    Rubrics:
                    
                    coordinate system:
                    The below mentioned is the coordinate system of the drone:
                    +x --- left
                    -x  --- right 
                    +y --- backward
                    -y  --- forward
                    +z --- up
                    If you reduce the value from +z --- down. 
                    0 is ground
                    the tuple is like [x,y,z]

                    operation_list:
                    square

                    Guidlines:
                    If the height at which the drone has to fly is not mentioned then take it default as 5 meters 
                    Once it is the end of the operation wait for next response from user
                    If the user mentions anything that's not from operation_list try to find the closest synonym of word from the operation_list, or else send a output which says "command not understood, please make it more clear"

                    
                    Examples:

                    Operation: square
                    If the user mentions about a square of some particular units or area. 
                    input: square of 5 units
                    Your output should be:
                    [0,0,5] 
                    [0,-5,5] 
                    [-5,-5,5] 
                    [-5,0,5]
                    [0,0,5] 
                    Explaination:
                    First the drone initial position will be [0,0,0]. Since the height is not mentioned take the default height mentioned in guidlines, so we need to make it go up by 5 units, so we increase the z by 5 units according to coordinate system to go up. this generates the first line [0,0,5]
                    After this we need to make it go forward for 5 meters at the same height so in addition to the first line which is [0,0,5] we change the y coordinate by decreasing it by 5 units according to coordinate system to [0,-5,5], this will be the second line of output
                    After this we need to make it go right for 5 meters at the same height so in addition to the second line which is [0,-5,5] we change the x coordinate by decreasing it by 5 units according to coordinate system to [-5,-5,5], this will be the third line of output
                    After this we need to make it go backward for 5 meters at the same height so in addition to the third line which is [-5,-5,5] we change the y coordinate by increasing it by 5 units according to coordinate system to [-5,0,5], this will be the fourth line of output
                    After this we need to make it go left for 5 meters at the same height so in addition to the fourth line which is [-5,0,5] we change the x coordinate by increasing it by 5 units according to coordinate system to [0,0,5], this will be the fifth line of output
            
                    
                    
                   
                    Conversation Output Format:
                    The outcome of your interactions must be documented as follows, without using markdown script for responses. Please only return a json object with:
                    1. "text": Waypoints deployed on the drone
                    2. "coordinates": The output you have been trained to generate

                    """

        
    def generate_response(self, prompt):
        with user():
            self.llm_chain += f"""\
                                {prompt}"""

        with assistant():
            self.llm_chain += gen(name='answer', max_tokens=500)
        
        return self.llm_chain
    

class DroneControl(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

        # List to store history of prompts and responses
        self.history = []

        # ROS node initialization
        rospy.init_node('drone_controller')
        self.publisher = rospy.Publisher('/drone4/mavros/setpoint_position/local', PoseStamped, queue_size=10)
        self.subscriber=rospy.Subscriber('/drone4/mavros/local_position/pose',PoseStamped,self.current_position_callback)
        #this will be updated using llm
        self.initial=None
        self.setpoints=None
        self.currentposition=None
        self.desired_orientation = Quaternion(x=0.0, y=0.0, z=0.7071, w=0.7071)

    def current_position_callback(self,data):
        self.currentposition=data.pose

        if(self.initial is None):
            self.initial=data.pose

    def init_ui(self):
        # Set window properties
        self.setWindowTitle('Drone Control')
        self.resize(800, 600)  # Set initial size of the window

        # Text fields for prompt, response, and history
        self.text_edit_prompt = QTextEdit()
        self.text_edit_response = QTextEdit()
        self.text_edit_response.setReadOnly(True)
        self.text_edit_history = QTextEdit()
        self.text_edit_history.setReadOnly(True)

        # Labels
        label_prompt = QLabel("Enter command:")
        label_response = QLabel("Response:")
        label_history = QLabel("History:")

        # Button to generate response
        self.button_generate = QPushButton('Generate Response')
        self.button_generate.clicked.connect(self.generate_response)

        # Button to clear history
        self.button_clear = QPushButton('Clear History')
        self.button_clear.clicked.connect(self.clear_history)

        # Button to execute controller
        self.button_execute = QPushButton('Execute Controller')
        self.button_execute.clicked.connect(self.execute_controller)

        # Apply some basic styles
        self.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                margin-bottom: 5px;
            }
            QTextEdit, QPushButton {
                font-size: 12px;
            }
        """)

        # Layout for labels
        layout_labels = QVBoxLayout()
        layout_labels.addWidget(label_prompt)
        layout_labels.addWidget(label_response)
        layout_labels.addWidget(label_history)

        # Layout for text fields
        layout_texts = QVBoxLayout()
        layout_texts.addWidget(self.text_edit_prompt)
        layout_texts.addWidget(self.text_edit_response)
        layout_texts.addWidget(self.text_edit_history)

        # Horizontal layout for text fields and labels
        layout_input_output = QHBoxLayout()
        layout_input_output.addLayout(layout_labels)
        layout_input_output.addLayout(layout_texts)

        # Vertical layout for buttons and input/output layout
        layout_controls = QVBoxLayout()
        layout_controls.addWidget(self.button_generate)
        layout_controls.addWidget(self.button_clear)
        layout_controls.addWidget(self.button_execute)
        layout_controls.addLayout(layout_input_output)

        # Main layout
        layout = QHBoxLayout()
        layout.addLayout(layout_controls)
        self.setLayout(layout)

    def generate_response(self):
        # Get the prompt text
        prompt_text = self.text_edit_prompt.toPlainText()
        gpt = llm()
        gpt.init_llm()
        json_val=gpt.generate_response(prompt_text)['answer']
        result = json.loads(json_val)
        self.setpoints=result["coordinates"]
        #self.setpoints=[[0,0,10],[0,-5,10]]


        #self.setpoints=[[0, 0, 5], [0, 5, 5], [1, 5, 5], [2, 5, 5], [3, 5, 5], [4, 5, 5], [4, 4, 5], [5, 4, 5], [5, 3, 5], [5, 2, 5], [5, 1, 5], [5, 0, 5], [5, -1, 5], [5, -2, 5], [5, -3, 5], [5, -4, 5], [4, -4, 5], [4, -5, 5], [3, -5, 5], [2, -5, 5], [1, -5, 5], [0, -5, 5], [-1, -5, 5], [-2, -5, 5], [-3, -5, 5], [-4, -5, 5], [-4, -4, 5], [-5, -4, 5], [-5, -3, 5], [-5, -2, 5], [-5, -1, 5], [-5, 0, 5], [-5, 1, 5], [-5, 2, 5], [-5, 3, 5], [-5, 4, 5], [-4, 4, 5], [-4, 5, 5], [-3, 5, 5], [-2, 5, 5], [-1, 5, 5], [0, 5, 5], [0, 0, 5]]
        # Generate response (replace this with your actual ChatGPT code)
        response_text = "Sample response to the command: '" + prompt_text + "\n'"
        response_text+=str(result)
        # Update response text field
        self.text_edit_response.setPlainText(response_text)

        # Update history
        self.history.append((prompt_text, response_text))
        self.update_history()

    def clear_history(self):
        # Clear the history list and update history text field
        self.history = []
        self.update_history()

    def update_history(self):
        # Display history in the history text field
        history_text = ""
        for prompt, response in self.history:
            history_text += f"Command: {prompt}\nResponse: {response}\n\n"
        self.text_edit_history.setPlainText(history_text)

    def execute_controller(self):
        pose=PoseStamped()
        #execute set points one by one untill it is reached
        #self.arm()
        print("starting")
        i=1
        for point in self.setpoints:
            print("waypoint no. : ",i)
            point=[point[0]+self.initial.position.x, point[1]+self.initial.position.y, point[2]+self.initial.position.z]
            pose.pose.position.x=point[0]
            pose.pose.position.y=point[1]
            pose.pose.position.z=point[2]

            desired_quaternion = (self.desired_orientation.x,
                                  self.desired_orientation.y,
                                  self.desired_orientation.z,
                                  self.desired_orientation.w)

            pose.pose.orientation.x=self.desired_orientation.x
            pose.pose.orientation.y=self.desired_orientation.y
            pose.pose.orientation.z=self.desired_orientation.z
            pose.pose.orientation.w=self.desired_orientation.w

            self.publisher.publish(pose)
            while True:
                dist=math.sqrt((self.currentposition.position.x-point[0])**2+(self.currentposition.position.y-point[1])**2+(self.currentposition.position.z-point[2])**2)
                current_quaternion = (self.currentposition.orientation.x,
                        self.currentposition.orientation.y,
                        self.currentposition.orientation.z,
                        self.currentposition.orientation.w)
                (roll, pitch, yaw) = euler_from_quaternion(current_quaternion)
                (desired_roll, desired_pitch, desired_yaw) = euler_from_quaternion(desired_quaternion)

                # You can set an acceptable threshold for orientation difference
                roll_diff = abs(roll - desired_roll)
                pitch_diff = abs(pitch - desired_pitch)
                yaw_diff = abs(yaw - desired_yaw)

                if(dist<0.5 and roll_diff < 0.1 and pitch_diff < 0.1 and yaw_diff < 0.1):
                    print("going to next")
                    break
            i+=1


if __name__ == '__main__':
    app = QApplication(sys.argv)
    drone_control = DroneControl()

    # Set window properties to fit the whole screen
    desktop = app.desktop()
    screen_rect = desktop.screenGeometry()
    #width, height = screen_rect.width(), screen_rect.height()
    #drone_control.resize(width, height)

    # Center the window on the screen
    window_rect = drone_control.frameGeometry()
    window_rect.moveCenter(screen_rect.center())
    drone_control.move(window_rect.topLeft())

    drone_control.show()
    sys.exit(app.exec_())
