#!/usr/bin/env nextflow

params.PROJECT_NAME = "tcga_tnbc"
params.PROJECT_VERSION = "tri"
params.resolution = "2"
r = params.resolution
params.y_interest = "LST_status"

// Folders
input_folder = "./outputs/${params.PROJECT_NAME}_${params.PROJECT_VERSION}"
output_folder = "${input_folder}/Model_NN_R${r}"

// labels
params.label_file = "/mnt/data4/tlazard/data/tcga_tnbc/labels_tcga_tnbc.csv"
label_file = file(params.label_file)

// Arguments
params.input_tiles = "${input_folder}/tiling/${r}/mat_pca/"
input_tiles = file(params.input_tiles)
params.mean_file = "${input_folder}/tiling/${r}/mean_pca/mean.npy"
mean_file = file(params.mean_file)
params.inner_fold = 5
inner_fold =  params.inner_fold
batch_size = 16
epochs = 40
repeat = 4
params.size = 5000
size = params.size
params.number_of_folds = 10
number_of_folds = params.number_of_folds 
params.model = "conan_a"
model = params.model

process Training_nn {
    publishDir "${output_model_folder}", pattern: "*.h5", overwrite: true
    publishDir "${output_results_folder}", pattern: "*.csv", overwrite: true
    memory { 30.GB + 5.GB * (task.attempt - 1) }
    errorStrategy 'retry'
    maxRetries 6
    cpus 5
    queue 'gpu-cbio'
    clusterOptions "--gres=gpu:1"
    // scratch true
    stageInMode 'copy'

    input:
    file path from input_tiles 
    each fold from 1..number_of_folds

    output:
    tuple val("${fold}"), file("*.csv") into results
    file("*.h5")

    script:
    python_script = file("./python/nn/main.py")
    output_model_folder = file("${output_folder}/${model}/${params.y_interest}/models/")
    output_results_folder = file("${output_folder}/${model}/${params.y_interest}/results/")

    /* Mettre --table --repeat --class_type en valeur par défaut ? */
    """
    module load cuda10.0
    python $python_script --mean_name $mean_file \
                          --path $path \
                          --table $label_file \
                          --batch_size $batch_size \
                          --epochs $epochs \
                          --size $size \
                          --fold_test $fold \
                          --repeat $repeat \
                          --y_interest $params.y_interest \
                          --inner_folds $inner_fold \
                          --model $model \
                          --workers 5
    """
}

