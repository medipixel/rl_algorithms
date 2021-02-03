# Using policy distillation


We implemented 3 featues for training policy distillation.

## 1. Student training using trained agent's data (expert data)

You can generate trained agent's data(expert data) by iterating the test episode.

```
python run_env_name.py --cfg-path <distillation-config-path> --load-from <teacher-checkpoint-path> --test 
```
The collected states will be stored in directory:  `data/distribution_buffer/<env_name>`.


If the expert data is generated, Put the path of the train-phase data in the dataset_path list in the distillation config file. Also change `is_student` to `True` in config file. And then execute the training just as the code below:

```
python run_env_name.py --cfg-path <distillation-config-path>  
```

You can set `epoch` and `batch_size` of the student learning through `epochs` and `batch_size` variables in the distillation config file.

## 2. Student training using training-phase states and trained agent 

This method provides the way to train the student using states that are generated as you train the agent(which we call it the train-phase data). 

Using distillation config file for training will automatically generate the train-phase data.
```
python run_env_name.py --cfg-path <distillation-config-path>
```

The generated data will be stored in directory:  `data/distribution_buffer/<env_name>`.


Since train-phase data doesn't contains the q value, you should load trained agent to generate q values for train-phase data. After putting the path of the train-phase data and changing `is_student` to `True` in the dataset_path list in the distillation config file, You can execute the training as the code below:
```
python run_env_name.py --cfg-path <distillation-config-path> --load-from <teacher-checkpoint-path>
```

## 3. Test student agent
If you only want to check the performance of the student agent, you should use the orginal agent config file instead of distillation config file. In pong environment for instance, you can use `dqn.py` config file instead of `distillation_dqn.py`. Using distillation config will also work well, but it will generate expert data while you're running the test. 
```
python run_env_name.py --test --load-from <student-checkpoint-path> --cfg-path <config-path>
```
