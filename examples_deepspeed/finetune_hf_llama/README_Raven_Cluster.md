# Tutorial: Fine-Tuning Llama2 using Megatron-Deepspeed

This tutorial will guide you through the process of fine-tuning Llama2.

## Creating singularity image

1- ```module load apptainer```

2- ```apptainer build --fakeroot images/megatron_06.sif container megatron.def```

```megatron_06``` indicates the version of singularity image in the project.

After running the above script, an image will be created in the images folder. Use this image for the following steps.

## Preprocessing

1- Download checkpoint from huggigface

For example, if you want to download llama2-7b, you can use the following script:

1- Make sure you are in ```Megatron-DeepSpeed``` directory

2- ```mkdir models/hf_model```

3- ```cd models/hf_model```

4- ```git lfs install```

5- ```git clone git@hf.co:meta-llama/Llama-2-7b```

After downloading the checkpoints of ```llama2-7b```, this model has two bin files named ```pytorch_model-00001-of-00002.bin``` and ```pytorch_model-00002-of-00002.bin```. However, for llama2-13b and llama2-70b, the number of .bin files is different. Remember the number of .bin files in this directory, which will be used in the following steps.


2- Convert HF checkpoints to megatron checkpoints

To convert the checkpoint, you should be familiar with the ```run_llama_7b.job``` script for further steps. Here is the explanation of this script:

```SBATCH --nodes=2```: This indicates the number of nodes that you want to use for converting checkpoints and fine tuning. For llama2-7b, 2 nodes is enough. 

It should be noted that this script can both convert checkpoints and fine-tune llama2-7b. Therefore, all information used during conversion should match that of the fine-tuning process. This implies that you will need to run this job twice: once for checkpoint conversion and once for fine-tuning. Further explanation will be provided later.

```DS_CONFIG```: Path of deepspeed config. It is recommended to use zero config 1.

```DATASET_PATH```: Path of dataset

```HF_LLAMA_PATH```: Path of huggiface model (downloded in the previous step)

```MEGA_DS_LLAMA_PATH```: Path of megatron model after conversion

```TP```: The size of tensor parallelization

```PP```: The size of pipeline parallelization

```ZERO_STAGE```: The zero stage that is using for deepspeed. This zero stage should be 1 (based on the ds config which is specified)

```MEGA_DS_LLAMA_PATH```: Path of megatron after conversion.

```HIDDEN_SIZE```: Hidden size
```FFN_HIDDEN_SIZE```: FFN hidden size
```NUM_LAYERS```: Number of layers. 
```NUM_HEADS```: Number of heads. 
```SEQ_LENGTH```: sequence length

The above information can be found in ```models/hf_model/Llama-2-7b-hf/config.json```.

To set fine tuning config, you can set in this section:

```MICRO_BATCH_SIZE=1```

```GLOBAL_BATCH_SIZE=8``` # e.g. llama: 4M tokens

```TRAIN_STEPS=250000``` # e.g. llama: 1T tokens / 4M tokens_per_batch = 250000 steps

```LR=3e-4```

```MIN_LR=3e-5```

```LR_WARMUP_STEPS=2000```

```WEIGHT_DECAY=0.1```

```GRAD_CLIP=1```

It should be noted that the ```SBATCH --nodes=2```, ```--gres=gpu:a100:4```, ```NUM_LAYERS```, and ```PP``` should be set to avoid error during converstion.

The number of layers (```NUM_LAYERS```) must be divisible by the size of ```PP``` (32/8), and the total number of GPUs must be divisible by the size of ```PP``` (8/8).

In the ```cat <<EOT > $DS_CONFIG```, The same DeepSpeed configuration should be specified.

```APPTAINER_ARGS```: To specify the apptainer argument, you can set different arguments here. The appropriate Singularity image should be set here.

```DISTRIBUTED_ARGS```: Distributed arguments.

```MODEL_ARGS```: In this section, parallelization, training arguments, model specifications, and other arguments should be specified.

To convert HF to megatron checkpoints, ```--hf-ckpt-num-shards``` in the ```HF_TO_MEGA_ARGS``` should be specified. As explained earlier, the number of .bin files in the hf model should be set here. For llama2-7b, the number of .bin files is 2.


## Converting huggigface checkpoints to Megatron checkpoints

If you follow the instruction and set all arguments correctly, you can convert huggigface checkpoints into megatron chechpoints using following script:

```sbatch run_llama_7b.job convert```

After conversion, in the directory of MEGA_DS_LLAMA_PATH, the new megatron model will be created and it will ready for fine tuing with 3D parallelization with Deepspeed.

It should be noted that during conversion, you need enough CPU memory to save megatron checkpoints. For example, 500G cpu memory is not enough for 70b model.

## Run fine tuning

After conversion, you can start fine tuning using:

```sbatch run_llama_7b.job```

## Tips for potential CUDA out-of-memory issue

After successful conversion, if you encounter CUDA out-of-memory errors during fine-tuning, you can consider increasing the number of GPUs, ```TP```, and ```PP```. Be mindful of divisibility related to the number of GPUs, ```PP```, and the number of model layers. Remember, each time you modify the job script, conversion should be performed before fine-tuning.













