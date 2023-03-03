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

mkdir -p features
mkdir -p features-count
mkdir -p features-normalized

# feature extraction from bitcode
cd sycl
for file in *.bc; do
    name="${file%.*}"

    echo "--- extracting features from sycl/$file ---"
    opt -load-pass-plugin ../libfeature_pass.so  --passes="print<feature>" -disable-output $file 1>> "../features/$name.temp"
done

cd ../features
for tempfile in *.temp; do
  name="${tempfile%.*}"

  features_count="../features-count/${name}_features.csv"
  features_norm="../features-normalized/${name}_features.csv"
  echo "kernel_name,mem_gl,int_add,int_bw,flt_mul,int_mul,flt_div,sp_fun,mem_loc,int_div,flt_add" > $features_count
  echo "kernel_name,mem_gl,int_add,int_bw,flt_mul,int_mul,flt_div,sp_fun,mem_loc,int_div,flt_add" > $features_norm

  lines_number=$(sed -n "/IR-function/=" $tempfile)

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
done

cd ..
rm -r features