

def assert_baseline(args) -> None:
    if args.word_embed == 'sbert':
        if not args.encoder == 'none':
            raise SystemExit("SentenceBERT only support 'none' encoder.")
        
    if args.encoder == 'transformer' or args.encoder == 'linear':
        if args.bidirectional:
            raise SystemExit("Bidirectionality only supported in LSTM or GRU encoder.")
        
    if args.encoder == 'lstm' or args.encoder == 'gru':
        if args.pooling != 'mean':
            raise SystemExit("No pooling can be performed for RNN-based models.")