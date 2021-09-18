conda activate PartAssembly
python ./gen_shape_pipeline  \
    --source_dir syn_chair_linspace \
    --gen_info {"Chair": (3,3,3,3,3,3,3)} \
    --method linspace
    --num_core 32
