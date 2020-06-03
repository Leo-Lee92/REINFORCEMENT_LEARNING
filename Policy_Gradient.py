# %%
import tensorflow as tf
from tensorflow import keras 
import gym 
import matplotlib
import matplotlib.pyplot as plt
import time
import numpy as np
# %%
# env = gym.make("CartPole-v1") # gym.make("환경 명") : 환경을 선언해준다. 
# obs = env.reset() # 환경을 초기화해준다.
# obs


# %%
env = gym.make("CartPole-v1") # gym.make("환경 명") : 환경을 호출해준다.
env.action_space # 호출된 환경 (현재는 "CartPole-v1")에서 가능한 행위공간을 보여준다.
                 # Discrete(2)는 가능한 행동이 0, 1 두 가지 정수임을 의미한다.
                 # 

# %%
# ! main() 메인함수가 정의된 script block은 반드시 python script를 실행하여 구동하여야 한다.
# ! 그 이유는 Jupyter Notebook, vs-code 등 프로그래밍 익스텐션 내에는 
# ! 그래픽 라이브러리 

# 정책함수 정의
def basic_policy(obs):
    angle = obs[2] # obs[2]는 카트 위 막대의 각도 (angle)에 대한 수치이다. 
                   # 막대가 수직일 시 obs[2] = 0
                   # 왼쪽으로 기울어질 시 obs[2] = 음수
                   # 오른쪽으로 기울어질 시 obs[2] = 양수

                   # 본 정책함수는 보상 (reward)인자가 활용되지 않으므로 
                   # 보상을 '학습'이 강화되지 않는 정책이라고 할 수 있음.

    return 0 if angle < 0 else 1 # obs[2]가 음수이면 0을 반환, 양수이면 1을 반환한다.

# 메인 함수 (python script 실행시 호출되는 함수)
def main():
    env = gym.make("CartPole-v1") # gym.make("환경 명") : 환경을 선언해준다. 
    totals = [] # 각 에피소드 벌 평균 보상을 담을 변수 totals 정의
       
    for episode in range(500): # 에피소드 500회 실행
        episode_rewards = 0 # 에피소드 보상 초기화
        obs = env.reset() # 환경 초기화
                          # env.reset()이 반환하는 obs값은 크기가 4인 1D 배열이다.
                          # 배열 내 4개의 변수는 각각 카트위치, 카트속도, 막대 (폴)각도, 막대 (폴)각속도를 의미한다.
                          ## "속도"는 속력 x 방향으로 카트속도, 막대각속도는 속력과 방향 정보를 모두 내포한다.

        print('episode : ', episode) # 몇 회째 에피소드 진행중인가 출력

        while True: # 각 에피소드 별로 env.step 무한번 실행
            action = basic_policy(obs) # 정책함수로부터 이전 obs[2]가 양수냐 음수냐에 따라 
                                       # 0 혹은 1을 반환하여 다음 행위(action)를 정의

            obs, reward, done, info = env.step(action) # env.step() 함수는 action을 인자로 받는다.
                                                       # env.step()은 인자로 받은 action을 수행했을 때 
                                                       # 다음 상태 (obs), 보상 (reward), 에피소드 엔딩여부 (done) 등을 반환한다.

            env.render() # env.render()는 새로운 상태가 진행된 현재 환경을 렌더링한다.

            episode_rewards += reward # 각 에피소드 별 보상 (reward)의 총합을 계산한다.

            if done: # 에피소드 종료 여부가 True라면 현 에피소드를 종료하고 다음 에피소드로 넘어간다.
                break

        totals.append(episode_rewards) # 각 에피소드별 보상 총합을 totals라는 리스트에 담는다.
        print(np.mean(totals), np.std(totals), np.min(totals), np.max(totals)) # 모든 에피소드의 보상 총합의 누적 평균을 계산한다.

        env.close() # 환경을 종료하고 창을 닫는다.

# 실행구조 작성
if __name__ == "__main__": # __name__ 은 python에서 내부적으로 사용하는 특별한 변수이다. python script.py 파일을 실행할 때 __name__ 변수에 " __main__" 값이 자동으로 할당된다.
                           # 반면 import script.py를 실행할 때는 __name__ 변수에 "script"가 할당된다. 즉 script.py을 실행하지 않고 안에 작성된 모듈들만 활성화 (import) 할 수 있다.
                           # 요약하자면, 모듈 활성화와 실행을 구분하기 위해 if __name__ = "__main__" 구문을 마지막에 추가한다고 생각하면 된다.

    main()  # 본 script 파일을 실행하면 (python Policy_Gradient.py) main() 함수를 호출하여라.


# %%

# %%
