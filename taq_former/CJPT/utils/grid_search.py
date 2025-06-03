
import itertools

def grid_search_weights(cfg, ttcl, tctg, fttm, table_encoder, text_encoder, qformer, train_loader, valid_loader):
    candidates = [0.1, 0.3, 0.5, 0.7, 1.0]
    best_score = 0
    best_weights = {}

    for weights in itertools.product(candidates, repeat=3):
        w_ttcl, w_tctg, w_fttm = weights
        val_score = 0.0
        if val_score > best_score:
            best_score = val_score
            best_weights = {'ttcl': w_ttcl, 'tctg': w_tctg, 'fttm': w_fttm}

    return best_weights

