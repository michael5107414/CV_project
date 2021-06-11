#!/bin/bash

> $1
echo "----------------------------------------------------------------" >> $1
start=`date +%s`
for i in {0..6}
do
	python eval.py --gt data/validation/0_center_frame/${i}/GT/frame10i11.png --cmp result/validation/0_center_frame/${i}/frame10i11.jpg >> $1
done
end=`date +%s`

echo "----------------------------------------------------------------" >> $1
echo "task    1   spend $((end - start))  sec[s]"
echo "----------------------------------------------------------------" >> $1

for i in {0..2}
do
	start=`date +%s`
	for j in {0..11}
	do
		for k in $(seq -f "%05g" $((j*8+1)) $((j*8+7)))
		do
			python eval.py --gt data/validation/1_30fps_to_240fps/${i}/${j}/GT/${k}.jpg --cmp result/validation/1_30fps_to_240fps/${i}/${j}/${k}.jpg >> $1
		done
	done
	end=`date +%s`
	echo "----------------------------------------------------------------" >> $1
	echo "subtask 2-${i} spend $((end - start)) sec[s]"
done

echo "----------------------------------------------------------------" >> $1

for i in {0..2}
do
	start=`date +%s`
	for j in {0..7}
	do
		for k in $(seq -f "%05g" $((j*10/4*4+4)) 4 $((j*10/4*4+8)))
		do
			python eval.py --gt data/validation/2_24fps_to_60fps/${i}/${j}/GT/${k}.jpg --cmp result/validation/2_24fps_to_60fps/${i}/${j}/${k}.jpg >> $1
		done
	done
	end=`date +%s`
	echo "----------------------------------------------------------------" >> $1
	echo "subtask 3-${i} spend $((end - start)) sec[s]"
done