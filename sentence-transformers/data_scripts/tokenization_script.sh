NSPLIT=128 #Must be larger than the number of processes used during training
FILENAME=en_XX.txt
INFILE=./${FILENAME}
TOKENIZER=bert-base-uncased
#TOKENIZER=bert-base-multilingual-cased
SPLITDIR=./tmp-tokenization-${TOKENIZER}-${FILENAME}/
OUTDIR=./encoded-data/${TOKENIZER}/$(echo "$FILENAME" | cut -f 1 -d '.')
NPROCESS=8

mkdir -p ${SPLITDIR}
echo ${INFILE}
split -a 3 -d -n l/${NSPLIT} ${INFILE} ${SPLITDIR}

pids=()

for ((i=0;i<$NSPLIT;i++)); do
    num=$(printf "%03d\n" $i);
    FILE=${SPLITDIR}${num};
    #we used --normalize_text as an additional option for mContriever
    python3 preprocess.py --tokenizer ${TOKENIZER} --datapath ${FILE} --outdir ${OUTDIR} &
    pids+=($!);
    if (( $i % $NPROCESS == 0 ))
    then
        for pid in ${pids[@]}; do
            wait $pid
        done
    fi
done

for pid in ${pids[@]}; do
    wait $pid
done

echo ${SPLITDIR}

rm -r ${SPLITDIR}
