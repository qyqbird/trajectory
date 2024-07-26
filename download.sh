#!/bin/bash

download_bigfile()
{
	path=`pwd`
	files=$(ls $path)
	for file in $files
	do
		if [[ $file == *bin ]];then
			echo $file
			git lfs fetch --include="$file"
		fi
	done
}

download_bigfile
