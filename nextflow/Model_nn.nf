#!/usr/bin/env nextflow

params.PROJECT_NAME = "biopsies_pet"
params.PROJECT_VERSION = "simCLR"
params.resolution = [1]
r = params.resolution
params.y_interest = "Residual"

// Folders
input_folder = "./outputs/${params.PROJECT_NAME}_${params.PROJECT_VERSION}"

// labels
params.label_file = "/mnt/data3/pnaylor/CellularHeatmaps/outputs/label_20_02_20.csv"
label_file = file(params.label_file)
input_tiles = file("${input_folder}/tiling/simCLR/1/mat/")
mean_file = file("${input_folder}/tiling/simCLR/1/mean/mean.npy")
// Arguments
params.inner_fold = 5
inner_fold =  params.inner_fold
batch_size = 16
epochs = 100 
repeat = 10
params.size = 1000
size = params.size
params.number_of_folds = 9
number_of_folds = params.number_of_folds 
params.model = "model_1S_a"
model = params.model
tiling_name = 'imagenet'

process Training_nn {
    publishDir "${output_model_folder}", pattern: "*.h5", overwrite: true, mode: 'copy'
    publishDir "${output_results_folder}", pattern: "*.csv", overwrite: true, mode: 'copy'
    memory { 30.GB + 5.GB * (task.attempt - 1) }
    errorStrategy 'retry'
    maxRetries 6
    cpus 5
    maxForks 4
    queue 'gpu-cbio'
    clusterOptions "--gres=gpu:1"
    // scratch true
    stageInMode 'copy'

    input:
    val r from params.resolution
    each fold from 1..number_of_folds

    output:
    tuple val("${fold}"), file("*.csv") into results
    file("*.h5")

    script:
    input_tiles = file("${input_folder}/tiling/${tiling_name}/${r}/mat_pca/")
    mean_file = file("${input_folder}/tiling/${tiling_name}/${r}/mean_pca/mean.npy")
    python_script = file("./python/nn/main.py")
    output_folder = "${input_folder}/Model_nn_R${r}_depth_256/"
    output_model_folder = file("${output_folder}/${model}/models/")
    output_results_folder = file("${output_folder}/${model}/results/")

    /* Mettre --table --repeat --class_type en valeur par défaut ? */
    """
    module load cuda10.0
    python $python_script --mean_name $mean_file \
                          --path $input_tiles \
                          --table $label_file \
                          --batch_size $batch_size \
                          --epochs $epochs \
                          --size $size \
                          --fold_test $fold \
                          --repeat $repeat \
                          --y_interest $params.y_interest \
                          --inner_folds $inner_fold \
                          --model $model \
                          --workers 5 \
                          --input_depth 256
    """
}

    
