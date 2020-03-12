nextflow run nextflow/Tiling-encoding.nf -resume -c ~/.nextflow/config -profile mines \
                                         --tiff_location /mnt/data4/tlazard/data/tcga_tnbc/images/ \
                                         --tissue_bound_annot /mnt/data4/tlazard/data/tcga_tnbc/annotations/annotations_tcga_tnbc_tristan/ \ 
                                         --PROJECT_NAME tcga_tnbc \
                                         --PROJECT_VERSION tri 