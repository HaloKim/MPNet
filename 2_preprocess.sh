fairseq-preprocess \
    --only-source \
    --srcdict dict.txt \
    --trainpref train.bpe \
    --validpref valid.bpe \
    --testpref test.bpe \
    --destdir wikitext-103 \
    --workers 60

