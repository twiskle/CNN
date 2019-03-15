#!/bin/bash
# Created by: Pchu (Oct 2018)

#this script is used to automate run the following:
#model4_test_segmented.py:      running the layered CNN for each kfold sequence

clear

BASEDIR=$(dirname "$0")
echo "script directory: '$BASEDIR'"

random_state=7
echo "random_state"
echo ${random_state}

#forloop step
step=0
#step=5

while [ $step -le 5 ]
do
    echo "*****step number*******"
    echo $step
    #python "$BASEDIR"/model4_test_segmented.py ${random_state} ${step}
    #python "$BASEDIR"/model4b_test_segmented_with_MM.py ${random_state} ${step}
    #python "$BASEDIR"/model5.py ${random_state} ${step}
    #python "$BASEDIR"/model5_one-param_nsplit6_1a.py ${random_state} ${step}
    #python "$BASEDIR"/model6_image_gen.py ${random_state} ${step}
    #python "$BASEDIR"/model6_image_gen_test2d.py ${random_state} ${step}
    #python "$BASEDIR"/model6_image_gen_test3d.py ${random_state} ${step}
    python "$BASEDIR"/model5_one-param_autoencode_with-gaus-noise.py ${random_state} ${step}
    ((step++))
done

echo "All DONE"