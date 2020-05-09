import pickle
import numpy as np
from mlgame.communication import ml as comm
import os.path as path

def ml_loop(side: str):
    """
    The main loop for the machine learning process
    The `side` parameter can be used for switch the code for either of both sides,
    so you can write the code for both sides in the same script. Such as:
    ```python
    if side == "1P":
        ml_loop_for_1P()
    else:
        ml_loop_for_2P()
    ```
    @param side The side which this script is executed for. Either "1P" or "2P".
    """

    # === Here is the execution order of the loop === #
    # 1. Put the initialization code here
    ball_served = False
    filename1p = path.join(path.dirname(__file__),"save","clf_SVC_pingpong1p.pickle")
    filename2p = path.join(path.dirname(__file__),"save","clf_SVC_pingpong2p.pickle")
    with open(filename1p, 'rb') as file1p:
        clf1p = pickle.load(file1p)
    with open(filename2p, 'rb') as file2p:
        clf2p = pickle.load(file2p)


    # 2. Inform the game process that ml process is ready before start the loop.
    scene_info = comm.recv_from_game()
    
    s = [93,93]
    def get_direction(VectorX,VectorY):
        if(VectorX>=0 and VectorY>=0):
            return 0
        elif(VectorX>0 and VectorY<0):
            return 1
        elif(VectorX<0 and VectorY>0):
            return 2
        elif(VectorX<0 and VectorY<0):
            return 3
        else:
            return 4
    # 3. Start an endless loop.
    while True:
        feature = []
        feature.append(sceneInfo['ball'][0])
        feature.append(sceneInfo['ball'][1])
        feature.append(sceneInfo['platform_1P'][0])
        feature.append(sceneInfo['platform_1P'][1])
        feature.append(sceneInfo['platform_2P'][0])
        feature.append(sceneInfo['platform_2P'][1])
        feature.append(sceneInfo['ball_speed'][0])
        feature.append(sceneInfo['ball_speed'][1])
        feature.append(get_direction(sceneInfo['ball_speed'][0],sceneInfo['ball_speed'][1]))

        feature = np.array(feature)

        # 3.2. If either of two sides wins the game, do the updating or
        #      resetting stuff and inform the game process when the ml process
        #      is ready.
        if scene_info["status"] != "GAME_ALIVE":
            # Do some updating or resetting stuff
            ball_served = False

            # 3.2.1 Inform the game process that
            #       the ml process is ready for the next round
            comm.ml_ready()
            continue

        if not ball_served:
            comm.send_to_game({"frame": scene_info["frame"], "command": "SERVE_TO_LEFT"})
            ball_served = True
        else:
            if side == "1P":
                command = clf1p.predict(feature)
            else:
                command = clf2p.predict(feature)

            if command == 0:
                comm.send_to_game({"frame": scene_info["frame"], "command": "NONE"})
            elif command == 1:
                comm.send_to_game({"frame": scene_info["frame"], "command": "MOVE_RIGHT"})
            else :
                comm.send_to_game({"frame": scene_info["frame"], "command": "MOVE_LEFT"})
