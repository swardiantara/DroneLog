

def assert_baseline(args) -> None:
    if args.word_embed == 'sbert':
        if not args.encoder == 'none':
            raise SystemExit("SentenceBERT only support 'none' encoder.")
        
    if args.encoder == 'transformer' or args.encoder == 'none':
        if args.bidirectional:
            raise SystemExit("Bidirectionality is only supported for LSTM or GRU encoder.")
        
        if args.pooling == 'last':
            raise SystemExit("Pooling 'last' is only supported for LSTM or GRU encoder.")
    else:
        if args.pooling == 'cls':
            raise SystemExit("Pooling CLS cannot be used for LSTM or GRU encoder.")
    
    if args.exclude_cls_before == True and args.pooling == 'cls':
        raise SystemExit("Pooling CLS can only be used when exclude_cls_before is False.")
    
    if args.exclude_cls_after == True and args.pooling == 'cls':
        raise SystemExit("Pooling CLS can only be used when exclude_cls_after is False.")        
    
               


def assert_multihead(args) -> None:               
    if args.label_schema_1 == args.label_schema_2:
        raise SystemExit("label_schema_1 cannot be the same as label_schema_2")            
    else:
        if not ((args.label_schema_1 == "binary" and args.label_schema_2 == "one_hot") or (args.label_schema_1 == "binary" and args.label_schema_2 == "101") or (args.label_schema_1 == "binary" and args.label_schema_2 == "111") or (args.label_schema_1 == "one_hot" and args.label_schema_2 == "101") or (args.label_schema_1 == "one_hot" and args.label_schema_2 == "111") or (args.label_schema_1 == "101" and args.label_schema_2 == "111")):
            raise SystemExit("The label schema combination is not valid!")            
    if (args.label_schema_1 == 'binary' and args.loss_1 != 'bce') or (args.label_schema_2 == 'binary' and args.loss_2 != 'bce'):
        raise SystemExit("BCE is the only valid loss function when label_schema == 'binary'")            
    if args.label_schema_1 == 'one_hot':
        if not (args.loss_1 == 'cross_entropy' or args.loss_1 == 'focal'):
            raise SystemExit("Please use cross_entropy or focal loss function when label_schema == 'one_hot'")            
    if args.label_schema_2 == 'one_hot':
        if not (args.loss_2 == 'cross_entropy' or args.loss_2 == 'focal'):
            raise SystemExit("Please use cross_entropy or focal loss function when label_schema == 'one_hot'")            
    if args.label_schema_1 == '101' or args.label_schema_1 == '111':
        raise SystemExit("Multitask label is not supported yet!")
        if args.loss_1 != 'logloss':
            raise SystemExit("Please use logloss loss function when label_schema == '101' or '111'")            
    if args.label_schema_2 == '101' or args.label_schema_2 == '111':
        raise SystemExit("Multitask label is not supported yet!")
        if args.loss_2 != 'logloss':
            raise SystemExit("Please use logloss loss function when label_schema == '101' or '111'")
    # if args.encoder == 'lstm' or args.encoder == 'gru':
    #     if args.pooling != 'mean':
    #         raise SystemExit("No pooling can be performed for RNN-based models.")