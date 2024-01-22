import os
import textwrap
import time
from rich import print
from typing import List

from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.callbacks import get_openai_callback, OpenAICallbackHandler, StreamingStdOutCallbackHandler

from dilu.scenario.envScenario import EnvScenario


delimiter = "####"
example_message = textwrap.dedent(f"""\
        {delimiter} Driving scenario description:
        You are driving on a road with 4 lanes, and you are currently driving in the second lane from the left. Your speed is 25.00 m/s, acceleration is 0.00 m/s^2, and lane position is 363.14 m. 
        There are other vehicles driving around you, and below is their basic information:
        - Vehicle `912` is driving on the same lane of you and is ahead of you. The speed of it is 23.30 m/s, acceleration is 0.00 m/s^2, and lane position is 382.33 m.
        - Vehicle `864` is driving on the lane to your right and is ahead of you. The speed of it is 21.30 m/s, acceleration is 0.00 m/s^2, and lane position is 373.74 m.
        - Vehicle `488` is driving on the lane to your left and is ahead of you. The speed of it is 23.61 $m/s$, acceleration is 0.00 $m/s^2$, and lane position is 368.75 $m$.

        {delimiter} Your available actions:
        IDLE - remain in the current lane with current speed Action_id: 1
        Turn-left - change lane to the left of the current lane Action_id: 0
        Turn-right - change lane to the right of the current lane Action_id: 2
        Acceleration - accelerate the vehicle Action_id: 3
        Deceleration - decelerate the vehicle Action_id: 4
        """)
example_answer = textwrap.dedent(f"""\
        Well, I have 5 actions to choose from. Now, I would like to know which action is possible. 
        I should first check if I can acceleration, then idle, finally decelerate.  I can also try to change lanes but with caution and not too frequently.

        - I want to know if I can accelerate, so I need to observe the car in front of me on the current lane, which is car `912`. The distance between me and car `912` is 382.33 - 363.14 = 19.19 m, and the difference in speed is 23.30 - 25.00 = -1.7 m/s. Car `912` is traveling 19.19 m ahead of me and its speed is 1.7 m/s slower than mine. This distance is too close and my speed is too high, so I should not accelerate.
        - Since I cannot accelerate, I want to know if I can maintain my current speed. I need to observe the car in front of me on the current lane, which is car `912`. The distance between me and car `912` is 382.33 - 363.14 = 19.19 m, and the difference in speed is 23.30 - 25.00 = -1.7 m/s. Car `912` is traveling 19.19 m ahead of me and its speed is 1.7 m/s slower than mine. This distance is too close and my speed is too high, so if I maintain my current speed, I may collide with it.
        - Maintain my current speed is not a good idea, so I can only decelearate to keep me safe on my current lane. Deceleraion is a feasible action.
        - Besides decelearation, I can also try to change lanes. I should carefully check the distance and speed of the cars in front of me on the left and right lanes. Noted that change-lane is not a frequent action, so I should not change lanes too frequently.
        - I first try to change lanes to the left. The car in front of me on the left lane is car `488`. The distance between me and car `488` is 368.75-363.14=5.61 m, and the difference in speed is 23.61 - 25.00=-1.39 m/s. Car `488` is traveling 5.61 m ahead of me and its speed is 1.39 m/s slower than mine. This distance is too close, the safety lane-change distance is 25m. Besides, my speed is higher than the front car on the left lane. If I change lane to the left, I may collide with it.                                           So I cannot change lanes to the left.
        - Now I want to see if I can change lanes to the right. The car in front of me on the right lane is car 864. The distance between me and car 864 is 373.74-363.14 = 10.6 m, and the difference in speed is 23.61-25.00=-3.7 m/s. Car 864 is traveling 10.6 m ahead of me and its speed is 3.7 m/s slower than mine. The distance is too close and my speed is higher than the front car on the right lane. the safety lane-change distance is 25m. if I change lanes to the right, I may collide with it. So I cannot change lanes to the right.
        - Now my only option is to slow down to keep me safe.
        Final Answer: Deceleration
                                         
        Response to user:#### 4
        """)


class DriverAgent:
    def __init__(
        self, sce: EnvScenario,
        temperature: float = 0, verbose: bool = False
    ) -> None:
        self.sce = sce
        oai_api_type = os.getenv("OPENAI_API_TYPE")
        if oai_api_type == "azure":
            print("Using Azure Chat API")
            self.llm = AzureChatOpenAI(
                callbacks=[
                    OpenAICallbackHandler()
                ],
                deployment_name=os.getenv("AZURE_CHAT_DEPLOY_NAME"),
                temperature=temperature,
                max_tokens=2000,
                request_timeout=60,
                streaming=True,
            )
        elif oai_api_type == "openai":
            print("Use OpenAI API")
            self.llm = ChatOpenAI(
                temperature=temperature,
                callbacks=[
                    OpenAICallbackHandler()
                ],
                model_name=os.getenv("OPENAI_CHAT_MODEL"),
                max_tokens=2000,
                request_timeout=60,
                streaming=True,
            )

    def few_shot_decision(self, scenario_description: str = "Not available", previous_decisions: str = "Not available", available_actions: str = "Not available", driving_intensions: str = "Not available", fewshot_messages: List[str] = None, fewshot_answers: List[str] = None):
        # for template usage refer to: https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/

        system_message = textwrap.dedent(f"""\
        You are ChatGPT, a large language model trained by OpenAI. Now you act as a mature driving assistant, who can give accurate and correct advice for human driver in complex urban driving scenarios.
        You will be given a detailed description of the driving scenario of current frame along with your history of previous decisions. You will also be given the available actions you are allowed to take. All of these elements are delimited by {delimiter}.

        Your response should use the following format:
        <reasoning>
        <reasoning>
        <repeat until you have a decision>
        Response to user:{delimiter} <only output one `Action_id` as a int number of you decision, without any action name or explanation. The output decision must be unique and not ambiguous, for example if you decide to decelearate, then output `4`> 

        Make sure to include {delimiter} to separate every step.
        """)

        human_message = f"""\
        Above messages are some examples of how you make a decision successfully in the past. Those scenarios are similar to the current scenario. You should refer to those examples to make a decision for the current scenario. 

        Here is the current scenario:
        {delimiter} Driving scenario description:
        {scenario_description}
        {delimiter} Driving Intensions:
        {driving_intensions}
        {delimiter} Available actions:
        {available_actions}

        You can stop reasoning once you have a valid action to take. 
        """
        human_message = human_message.replace("        ", "")

        if fewshot_messages is None:
            raise ValueError("fewshot_message is None")
        messages = [
            SystemMessage(content=system_message),
            # HumanMessage(content=example_message),
            # AIMessage(content=example_answer),
        ]
        for i in range(len(fewshot_messages)):
            messages.append(
                HumanMessage(content=fewshot_messages[i])
            )
            messages.append(
                AIMessage(content=fewshot_answers[i])
            )
        messages.append(
            HumanMessage(content=human_message)
        )
        # print("fewshot number:", (len(messages) - 2)/2)
        start_time = time.time()
        # with get_openai_callback() as cb:
        # response = self.llm(messages)
        # print(response.content)
        print("[cyan]Agent answer:[/cyan]")
        response_content = ""
        for chunk in self.llm.stream(messages):
            response_content += chunk.content
            print(chunk.content, end="", flush=True)
        print("\n")
        decision_action = response_content.split(delimiter)[-1]
        try:
            result = int(decision_action)
            if result < 0 or result > 4:
                raise ValueError
        except ValueError:
            print("Output is not a int number, checking the output...")
            check_message = f"""
            You are a output checking assistant who is responsible for checking the output of another agent.
            
            The output you received is: {decision_action}

            Your should just output the right int type of action_id, with no other characters or delimiters.
            i.e. :
            | Action_id | Action Description                                     |
            |--------|--------------------------------------------------------|
            | 0      | Turn-left: change lane to the left of the current lane |
            | 1      | IDLE: remain in the current lane with current speed   |
            | 2      | Turn-right: change lane to the right of the current lane|
            | 3      | Acceleration: accelerate the vehicle                 |
            | 4      | Deceleration: decelerate the vehicle                 |


            You answer format would be:
            {delimiter} <correct action_id within 0-4>
            """
            messages = [
                HumanMessage(content=check_message),
            ]
            with get_openai_callback() as cb:
                check_response = self.llm(messages)
            result = int(check_response.content.split(delimiter)[-1])

        few_shot_answers_store = ""
        for i in range(len(fewshot_messages)):
            few_shot_answers_store += fewshot_answers[i] + \
                "\n---------------\n"
        print("Result:", result)
        return result, response_content, human_message, few_shot_answers_store
