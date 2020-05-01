#!/usr/bin/env nextflow

params.PROJECT_NAME = "luminaux_brca"
params.PROJECT_VERSION = "xml_mask"
output_folder = "./outputs/${params.PROJECT_NAME}_${params.PROJECT_VERSION}"

// Parameters of the python file
params.tiff_location = "/mnt/data4/tlazard/data/luminaux_BRCA/images" // tiff files to process DO NOT end with "/".
params.tissue_bound_annot = "/mnt/data4/tlazard/data/luminaux_BRCA/annotations/xml_guillaume" // DO NOT end with "/"
params.model_name = 'imagenet'
model_name = params.model_name
params.model_path = '/mnt/data4/tlazard/projets/simCLR/peter_biopsies/512/1/'
model_path = params.model_path
params.size = '224'
size = params.size
auto_mask = 0

// input file
tiff_files = file(params.tiff_location + "/*.ndpi")
boundaries_files = file(params.tissue_bound_annot)

//params.label = "/mnt/data3/pnaylor/CellularHeatmaps/outputs/label.csv"
//label = file(params.label)

// input parameter
params.weights = "imagenet"
weights = params.weights

params.innner_fold = 5
levels = [1, 2, 0]

process WsiTilingEncoding {
    publishDir "${output_process_mat}", overwrite: true, pattern: "${name}.npy", mode: 'copy'
    publishDir "${output_process_mean}", overwrite: true, pattern: "${name}_mean.npy", mode: 'copy'
    publishDir "${output_process_info}", overwrite: true, pattern: "${name}_info.txt", mode: 'copy'
    publishDir "${output_process_visu}", overwrite: true, pattern: "${name}_visu.png", mode: 'copy'

    queue "gpu-cbio"
    clusterOptions "--gres=gpu:1"
    maxForks 16
    memory '20GB'
    
    input:
    file slide from tiff_files
    each level from levels
    
    output:
    set val("$level"), file("${name}.npy") into bags
    set val("$level"), file("${name}_mean.npy") into mean_patient
    file("${name}_info.txt")
    file("${name}_visu.png")

    script:
    py = file("./python/preparing/process_one_patient.py")
    name = slide.baseName
    mask = boundaries_files + "/${name}"
    output_process_mean = "${output_folder}/tiling/${model_name}/${level}/mean"
    output_process_mat = "${output_folder}/tiling/${model_name}/${level}/mat"
    output_process_info = "${output_folder}/tiling/${model_name}/${level}/info"
    output_process_visu = "${output_folder}/tiling/${model_name}/${level}/visu"
    """
    module load cuda10.0
    python $py --slide $slide \
               --mask $mask \
               --analyse_level $level \
               --weight $weights \
               --model_name $model_name \
               --model_path $model_path \
               --size $size \
               --auto_mask $auto_mask
    """
}

mean_patient  .groupTuple() 
              .into { all_patient_means ; all_patient_means2 }


process ComputeGlobalMean {
    publishDir "${output_process}", overwrite: true, mode: 'copy'
    memory { 10.GB }
    input:
    set level, file(_) from all_patient_means
    output:
    file('mean.npy')

    script:
    compute_mean = file('./python/preparing/compute_mean.py')
    output_process = "${output_folder}/tiling/${model_name}/$level/mean/"

    """
    python $compute_mean 
    """
}

//y = ["Residual", "Prognostic"]
//process RandomForestlMean {
//   publishDir "${output_process}", overwrite: true
//   memory { 10.GB }
//   cpus 8
//   input:
//   set level, file(_) from all_patient_means2
//   file lab from label
//   each y_interest from y
//
//   output:
//   file('*.txt')
//
//   script:
//   compute_rf = file("./python/naive_rf/compute_rf.py")
//   output_process = "${output_folder}/naive_rf_${level}/${y_interest}"
//
//   """
//   python $compute_rf --label $label \
//                      --inner_fold $params.inner_fold \
//                      --y_interest $y_interest \
//                      --cpu 8
//   """
//}

// keep bags_1 to collect them and process the PCA
// bags_2 is a copy, to after compute the transformed tiles
// bags_per_level = [($level, (*.npy))], for each level the whole tiles files.
bags .into{ bags_1; bags_2 }
bags_1 .groupTuple()
     .set{ bags_per_level }

process Incremental_PCA {
    publishDir "${output_process_pca}", overwrite: true, mode: 'copy'
    memory '60GB'
    cpus '16'

    input:
    tuple level, files from bags_per_level
    
    output:
    tuple level, file("*.joblib") into results_PCA

    script:
    output_process_pca = "${output_folder}/tiling/${model_name}/${level}/pca/"
    input_tiles = file("${output_folder}/tiling/${model_name}/${level}/mat")
    python_script = file("./python/preparing/pca_partial.py")

    """
    python $python_script --path ${input_tiles}
    """
}

// files_to_transform = [ ($level, pca_res_level, f1.npy ), ($level, pca_res_level, f2.npy), ... ]
// begins to get filled as soon as a level has been treated by the PCA.
results_PCA .combine(bags_2, by: 0) 
            .set { files_to_transform } 

process Transform_Tiles {

    publishDir "${output_mat_pca}", overwrite: true, mode: 'copy'
    memory '20GB'

    input:
    tuple level, file(pca), tile from files_to_transform

    output:
    tuple level, file("*.npy") into transform_tiles

    script:
    output_mat_pca = "${output_folder}/tiling/${model_name}/$level/mat_pca"
    python_script = file("./python/preparing/transform_tile.py")

    """
    python ${python_script} --path $tile --pca $pca
    """
}

transform_tiles  .groupTuple() 
              .set { transform_tiles_per_level }

//<<<<<<< HEAD
//process ComputePCAGlobalMean {
//    publishDir "${output_pca_mean}", overwrite: true
//    memory { 10.GB }
//
//    input:
//    set level, file(_) from transform_tiles_per_level
//    output:
//    file('mean.npy')
//
//    script:
//    output_pca_mean = "${output_folder}/tiling/$level/pca_mean"
//    compute_mean_pca = file('./python/preparing/compute_mean_pca.py')
//
//    """
//    echo $level
//    python $compute_mean_pca
//    """
//}
//=======
process ComputePCAGlobalMean {
    publishDir "${output_pca_mean}", overwrite: true, mode: 'copy'
    memory { 10.GB }

    input:
    set level, file(_) from transform_tiles_per_level
    output:
    file('mean.npy')

    script:
    output_pca_mean = "${output_folder}/tiling/${model_name}/$level/pca_mean"
    compute_mean_pca = file('./python/preparing/compute_mean_pca.py')
    tiles_path = file("${output_folder}/tiling/${model_name}/$level/mat_pca")

    """
    echo $level
    python $compute_mean_pca --path $tiles_path
    """
}
