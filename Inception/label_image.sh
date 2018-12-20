#################################################
#	
#	Bash script to execute label.py, which labels an input using the train InceptionV3
#   model.
#
#	output_graph.pb, output_labels.txt, Placeholder and final_result are all from retrain.py
#   applied to the training data.
#
#	This script is called by classify_IE613_time.py, which attempt to classify radio bursts in
#  	the dynamic spectra from the ILOFAR station.
#
#	Written by Eoin Carley.
#
#################################################



dir=$1
python3 label_image.py --graph=/tmp/output_graph.pb --labels=/tmp/output_labels.txt --input_layer=Placeholder --output_layer=final_result --image=$dir/input.png > $dir/class_probs.txt
