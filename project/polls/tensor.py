from polls import demo_keyframe

class TensorFace():
    def __init__(self):
        class_names, best_class_indices, best_class_probabilities = demo_keyframe.main('C:/Users/mmlab/Desktop/FaceNetDemo/demo_1206/1sec')
        self.name = class_names[best_class_indices]
        self.score = best_class_probabilities
