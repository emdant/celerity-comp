#!/bin/bash
alias sycl="clang++ -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda -fsycl-device-only"
alias opt=opt-15
alias llvm-dis=llvm-dis-15

# SYCL compilation
for file in sycl/*.cpp; do
  echo "SYCL source: $file"
  name="${file%.*}"

  sycl $file -o $name.bc
  llvm-dis $name.bc
done

tempfile="temp_features.txt"
> $tempfile
# feature extraction from bitcode
for file in sycl/*.bc; do
    echo "--- extracing features from sycl/$file ---"
    opt -load-pass-plugin ./libfeature_pass.so  --passes="print<feature>" -disable-output $file 1>> $tempfile

    # ../features -i $bc 
    # ../features -i $bc -fe kofler
    # ../features -i $bc -fs full
done

features_count="features_count.csv"
features_norm="features_normalized.csv"
echo "kernel_name,mem_gl,int_add,int_bw,flt_mul,int_mul,flt_div,sp_fun,mem_loc,int_div,flt_add" > $features_count
echo "kernel_name,mem_gl,int_add,int_bw,flt_mul,int_mul,flt_div,sp_fun,mem_loc,int_div,flt_add" > $features_norm

lines_number=$(sed -n "/IR-function/=" temp_features.txt)

for n in $lines_number; do
  line_counters=""
  line_norm=""

  name="$(sed "${n}q;d" $tempfile | awk '{print $2}'),"
  line_counters+=$name
  line_norm+=$name

  counters_line_num=$(($n+6))
  counters=$(sed "${counters_line_num}q;d" $tempfile)
  let sum=0
  for num in $counters; do
    sum=$(( $sum + $num ))
  done
  if [ $sum -eq '0' ]; then
    continue
  fi
  line_counters+=$(echo $counters | sed 's/ /,/g')

  normalized_line_num=$(($n+8))
  normalized=$(sed "${normalized_line_num}q;d" $tempfile)
  line_norm+=$(echo $normalized | sed 's/ /,/g')

  echo $line_counters >> $features_count
  echo $line_norm >> $features_norm
done

rm $tempfile