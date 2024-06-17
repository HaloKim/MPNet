for SPLIT in train valid test; do \
    python3 ./MPNet/encode.py \
        --inputs /workspace/MPNet/${SPLIT}.txt \
        --outputs /workspace/MPNet/${SPLIT}.bpe \
        --keep-empty \
        --workers 60; \
done

