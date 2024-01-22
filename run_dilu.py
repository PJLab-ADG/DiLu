import copy
import random
import numpy as np
import yaml
import os
from rich import print

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from dilu.scenario.envScenario import EnvScenario
from dilu.driver_agent.driverAgent import DriverAgent
from dilu.driver_agent.vectorStore import DrivingMemory
from dilu.driver_agent.reflectionAgent import ReflectionAgent


test_list_seed = [5838, 2421, 7294, 9650, 4176, 6382, 8765, 1348,
                  4213, 2572, 5678, 8587, 512, 7523, 6321, 5214, 31]


def setup_env(config):
    if config['OPENAI_API_TYPE'] == 'azure':
        os.environ["OPENAI_API_TYPE"] = config['OPENAI_API_TYPE']
        os.environ["OPENAI_API_VERSION"] = config['AZURE_API_VERSION']
        os.environ["OPENAI_API_BASE"] = config['AZURE_API_BASE']
        os.environ["OPENAI_API_KEY"] = config['AZURE_API_KEY']
        os.environ["AZURE_CHAT_DEPLOY_NAME"] = config['AZURE_CHAT_DEPLOY_NAME']
        os.environ["AZURE_EMBED_DEPLOY_NAME"] = config['AZURE_EMBED_DEPLOY_NAME']
    elif config['OPENAI_API_TYPE'] == 'openai':
        os.environ["OPENAI_API_TYPE"] = config['OPENAI_API_TYPE']
        os.environ["OPENAI_API_KEY"] = config['OPENAI_KEY']
        os.environ["OPENAI_CHAT_MODEL"] = config['OPENAI_CHAT_MODEL']
    else:
        raise ValueError("Unknown OPENAI_API_TYPE, should be azure or openai")

    # environment setting
    env_config = {
        'highway-v0':
        {
            "observation": {
                "type": "Kinematics",
                "features": ["presence", "x", "y", "vx", "vy"],
                "absolute": True,
                "normalize": False,
                "vehicles_count": config["vehicle_count"],
                "see_behind": True,
            },
            "action": {
                "type": "DiscreteMetaAction",
                "target_speeds": np.linspace(5, 32, 9),
            },
            "lanes_count": 4,
            "other_vehicles_type": config["other_vehicle_type"],
            "duration": config["simulation_duration"],
            "vehicles_density": config["vehicles_density"],
            "show_trajectories": True,
            "render_agent": True,
            "scaling": 5,
            'initial_lane_id': None,
            "ego_spacing": 4,
        }
    }

    return env_config


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    config = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
    env_config = setup_env(config)

    REFLECTION = config["reflection_module"]
    memory_path = config["memory_path"]
    few_shot_num = config["few_shot_num"]
    result_folder = config["result_folder"]
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    with open(result_folder + "/" + 'log.txt', 'w') as f:
        f.write("memory_path {} | result_folder {} | few_shot_num: {} | lanes_count: {} \n".format(
            memory_path, result_folder, few_shot_num, env_config['highway-v0']['lanes_count']))

    agent_memory = DrivingMemory(db_path=memory_path)
    if REFLECTION:
        updated_memory = DrivingMemory(db_path=memory_path + "_updated")
        updated_memory.combineMemory(agent_memory)

    episode = 0
    while episode < config["episodes_num"]:
        # setup highway-env
        envType = 'highway-v0'
        env = gym.make(envType, render_mode="rgb_array")
        env.configure(env_config[envType])
        result_prefix = f"highway_{episode}"
        env = RecordVideo(env, result_folder, name_prefix=result_prefix)
        env.unwrapped.set_record_video_wrapper(env)
        seed = random.choice(test_list_seed)
        obs, info = env.reset(seed=seed)
        env.render()

        # scenario and driver agent setting
        database_path = result_folder + "/" + result_prefix + ".db"
        sce = EnvScenario(env, envType, seed, database_path)
        DA = DriverAgent(sce, verbose=True)
        if REFLECTION:
            RA = ReflectionAgent(verbose=True)

        response = "Not available"
        action = "Not available"
        docs = []
        collision_frame = -1

        try:
            already_decision_steps = 0
            for i in range(0, config["simulation_duration"]):
                obs = np.array(obs, dtype=float)

                print("[cyan]Retreive similar memories...[/cyan]")
                fewshot_results = agent_memory.retriveMemory(
                    sce, i, few_shot_num) if few_shot_num > 0 else []
                fewshot_messages = []
                fewshot_answers = []
                fewshot_actions = []
                for fewshot_result in fewshot_results:
                    fewshot_messages.append(
                        fewshot_result["human_question"])
                    fewshot_answers.append(fewshot_result["LLM_response"])
                    fewshot_actions.append(fewshot_result["action"])
                    mode_action = max(
                        set(fewshot_actions), key=fewshot_actions.count)
                    mode_action_count = fewshot_actions.count(mode_action)
                if few_shot_num == 0:
                    print("[yellow]Now in the zero-shot mode, no few-shot memories.[/yellow]")
                else:
                    print("[green4]Successfully find[/green4]", len(
                        fewshot_actions), "[green4]similar memories![/green4]")

                sce_descrip = sce.describe(i)
                avail_action = sce.availableActionsDescription()
                print('[cyan]Scenario description: [/cyan]\n', sce_descrip)
                # print('[cyan]Available actions: [/cyan]\n',avail_action)
                action, response, human_question, fewshot_answer = DA.few_shot_decision(
                    scenario_description=sce_descrip, available_actions=avail_action,
                    previous_decisions=action,
                    fewshot_messages=fewshot_messages,
                    driving_intensions="Drive safely and avoid collisons",
                    fewshot_answers=fewshot_answers,
                )
                docs.append({
                    "sce_descrip": sce_descrip,
                    "human_question": human_question,
                    "response": response,
                    "action": action,
                    "sce": copy.deepcopy(sce)
                })

                obs, reward, done, info, _ = env.step(action)
                already_decision_steps += 1

                env.render()
                sce.promptsCommit(i, None, done, human_question,
                                  fewshot_answer, response)
                env.unwrapped.automatic_rendering_callback = env.video_recorder.capture_frame()

                print("--------------------")

                if done:
                    print("[red]Simulation crash after running steps: [/red] ", i)
                    collision_frame = i
                    break
        finally:

            with open(result_folder + "/" + 'log.txt', 'a') as f:
                f.write(
                    "Simulation {} | Seed {} | Steps: {} | File prefix: {} \n".format(episode, seed, already_decision_steps, result_prefix))
                
            if REFLECTION:
                print("[yellow]Now running reflection agent...[/yellow]")
                if collision_frame != -1: # End with collision
                    for i in range(collision_frame, -1, -1):
                        if docs[i]["action"] != 4:  # not decelearate
                            corrected_response = RA.reflection(
                                docs[i]["human_question"], docs[i]["response"])
                            
                            choice = input("[yellow]Do you want to add this new memory item to update memory module? (Y/N): ").strip().upper()
                            if choice == 'Y':
                                updated_memory.addMemory(
                                    docs[i]["sce_descrip"],
                                    docs[i]["human_question"],
                                    corrected_response,
                                    docs[i]["action"],
                                    docs[i]["sce"],
                                    comments="mistake-correction"
                                )
                                print("[green] Successfully add a new memory item to update memory module.[/green]. Now the database has ", len(
                                    updated_memory.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.")
                            else:
                                print("[blue]Ignore this new memory item[/blue]")
                            break
                else:
                    print("[yellow]Do you want to add[/yellow]",len(docs)//5, "[yellow]new memory item to update memory module?[/yellow]",end="")
                    choice = input("(Y/N): ").strip().upper()
                    if choice == 'Y':
                        cnt = 0
                        for i in range(0, len(docs)):
                            if i % 5 == 1:
                                updated_memory.addMemory(
                                    docs[i]["sce_descrip"],
                                    docs[i]["human_question"],
                                    docs[i]["response"],
                                    docs[i]["action"],
                                    docs[i]["sce"],
                                    comments="no-mistake-direct"
                                )
                                cnt +=1
                        print("[green] Successfully add[/green] ",cnt," [green]new memory item to update memory module.[/green]. Now the database has ", len(
                                    updated_memory.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.")
                    else:
                        print("[blue]Ignore these new memory items[/blue]")
            

            print("==========Simulation {} Done==========".format(episode))
            episode += 1
            env.close()
