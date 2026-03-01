from collections import OrderedDict


class TranspositionTable:
    """
    Cache for previously evaluated positions.
    Stores position hash ? (depth, score, best_move, flag)
    """
    
    EXACT = 0
    LOWERBOUND = 1
    UPPERBOUND = 2
    
    def __init__(self, size_mb=256):
        """Initialize transposition table with size limit."""
        self.max_entries = (size_mb * 1024 * 1024) // 64  # Rough estimate
        self.table = OrderedDict()
    
    def store(self, zobrist_hash, depth, score, best_move, flag):
        """Store position evaluation."""
        if len(self.table) >= self.max_entries:
            # Remove oldest entry
            self.table.popitem(last=False)
        
        self.table[zobrist_hash] = {
            'depth': depth,
            'score': score,
            'best_move': best_move,
            'flag': flag
        }
    
    def probe(self, zobrist_hash, depth, alpha, beta):
        """
        Look up position in table.
        Returns (score, best_move) if usable, else (None, None)
        """
        if zobrist_hash not in self.table:
            return None, None
        
        entry = self.table[zobrist_hash]
        
        # Move to end (LRU)
        self.table.move_to_end(zobrist_hash)
        
        # Only use if searched to sufficient depth
        if entry['depth'] < depth:
            return None, entry['best_move']
        
        score = entry['score']
        flag = entry['flag']
        
        # Check if we can use this score
        if flag == self.EXACT:
            return score, entry['best_move']
        elif flag == self.LOWERBOUND and score >= beta:
            return score, entry['best_move']
        elif flag == self.UPPERBOUND and score <= alpha:
            return score, entry['best_move']
        
        return None, entry['best_move']
    
    def clear(self):
        """Clear the transposition table."""
        self.table.clear()