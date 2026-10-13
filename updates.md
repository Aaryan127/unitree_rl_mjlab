## Immediate TODOs
- [] Read https://arxiv.org/pdf/2010.11251 and understand the CPG implementation(What is F_i) @om update by [2-03-26] 
- [] Can we directly use Gait in Eight? @om
- [] Code the CPG for Unitree_mj 
- [] Reduce the height map res for efficiency
- [] Any other optimizations? - Training is slow.

## Todos
- [x] Look up the encoder arch in the supplemnentry section and change the teacher model - @asavari [23-02-26]
- [x] Are the rewards different - If yes, modify. - @chirag and @asavari - [23-02-26]
- [x] Check the current OnPoicyRunner implementation - @Om @Mrigaank - [23-02-26 EOD]
- [x] Sort the observations for their respective target networks - Sm prop. goes to actor, some goes to critic etc. -@om @mrigaank [23-02-26]
- [x] Go Through the rest of this updates.md - Use ipdb or pdb to inspect - @all [ideally parallely]
- [ ] Train and check - [24-02-26 HD]
- [-] Document all the work done so far - @all
- [x] Have individual heightmaps surrounding each leg - instead of a global grid.
## Heat Check after the above have been setup
- [] Inspect the observation space that goes as input to the network - Data type and Shape
- [] Are Critic/Actor Networks receiving the right inputs?
- [] The arch details(for the Teacher MLP/the encoder/decoder) fits the paper description as much as possible.

### MJX TODOs
- [] Port all of the above to Jax - @om and @Mrigaank


## Tracking updates
- [] Train the Teacher model 
- [] Update runner and configs to use [actorcritic_wild](mjlab/rsl_rl/modules/actor_critic_wild.py)
- [x] add Height map to obs: 21/2
- [x] seperate proprio and extero obs - seperated under different keys of obs dir: 21/2
- [x] write a teacher mlp that has extero and proprio encoder - Very basic network has been added
- [x] use teacher mlp in ActorCritic: Updated the actorcritic class to use teacher mlp which takes in proprio and extero seperately.


## Teacher Model Mods
- [] Encoders need to be figured out - TODOLATER
- [] Modify rsl_rl ppo and onpolicy runner to use actor_critic_wild.py
- [] Proprio, extero, noisy - best to keep separated, to keep forward pass simple.

## Observations
[] PROPRIOCETIVE OBSERVATIONS:
  [x]  Body vel - lin + ang [X]
  [x]  Orientation - [X]
  [x]  Joint position - [X]
  [x]  Velocty HISTORY - [Can be done from buffer, but then what is sent initially? Empty tensor - mp]
  [x]  ACTION HISTORY - [LAST ACTION AVAILABLE - GET HISTORY FROM BUFFER]
  []  LEG'S PHASE  -   [] 

[]  EXTEROCEPTIVE OBSERVATIONS:
  [x]  Heightmap arond the robot - [X]   5  DIFFERENT RADII [ignore for now]

[]  PRIVIELLGED OBSERVATIONS(CHECK REQUIRED):
  [x]  CONTACT STATES [X]
  [x]  CONTACT FORCES [X]
  [x]  CONTACT NORMALS [X]
  []  FRICTION COEFFICIENTS - CHange needed
  [x]  THIGH AND SHANK CONTACT STATES [X]
  [x]  EXTERNAL FORCES AND TORQUES ON THE BODY [X]

## Instructions on adding sensors:
STEP 0: Identify the right [task](mjlab/tasks).

STEP 1: add the sensor to the correct [env_config](mjlab/tasks/velocity/config/go2/env_cfgs.py)

STEP 2: add the sensor config to env and add that sensor to the seen, for sensor config look at [sensor](mjlab/sensor/__init__.py)

    - here make sure you are adding the sensor to the right frame
    - Unitree Go2 xml doesnt have frames for the sensors, some sensors simulate data wrt world while others need the right frame 
    - use visualise to look at the sensor data: if applicable

STEP 3: add the declare the sensor function in [observations](mjlab/tasks/velocity/mdp/observations.py), the sensor data is accessed using this function 

STEP 4: add the sensor function to the [observation file](mjlab/tasks/velocity/velocity_env_cfg.py), keep track of the sensor name declared in step 2.

## Instructions on adding observations from the robot 
STEP 0: Identify the right [task](mjlab/tasks).

STEP 1: Make a function in the [observations](mjlab/tasks/velocity/mdp/observations.py) that takes in the env and returns the required data from the robot entity

STEP 2: Add initialise the sensor at [task config file](mjlab/tasks/velocity/velocity_env_cfg.py) as a ObservationTerm

As it is not a real sensor but directly getting data from the bot, we do not need to add the sensor to the scene 

## Instructions on adding reward
STEP 0: Identify the right [task](mjlab/tasks).

STEP 1: Add reward function to [rewards](mjlab/tasks/velocity/mdp/rewards.py) 

NOTE: the expected and actual are of shape (batchsize, ), so the rewards have to be alloted keeping in the index in mind as each index correspond to differnt bots

STEP 2: add the reward config in [task config file](mjlab/tasks/velocity/velocity_env_cfg.py), declare the function and the parameters.

## Instructions on adding a Curriculum Scaled Reward Term

STEP 0: Identify the right [task](mjlab/tasks).

STEP 1: Add reward function to [rewards](mjlab/tasks/velocity/mdp/rewards.py). Make sure to:
- Return positive cost
- Do NOT include negative sign
- Do NOT include curriculum logic inside the function

STEP 2: Add the reward config in [task config file](mjlab/tasks/velocity/velocity_env_cfg.py) under rewards, declare the function and the parameters. **Set initial weight=0.0**

STEP 3: Add the reward config in [task config file](mjlab/tasks/velocity/velocity_env_cfg.py) under curriculum. Use the built-in reward_weight function. 
Example:
```
"my_reward_weight": CurriculumTermCfg(
    func=mdp.reward_weight,
    params={
        "reward_name": "my_reward",
        "weight_stages": [
            {"step": 0, "weight": 0.0},
            {"step": 2000 * 24, "weight": -0.1},
            {"step": 5000 * 24, "weight": -0.25},
        ],
    },
),
```

Note:
- reward_name must match the reward key
- weight corresponds to −cₖ
- Curriculum updates at environment reset

This is to add the curriculum term to scale the reward over epochs.
