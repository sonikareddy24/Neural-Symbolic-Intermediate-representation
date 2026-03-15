import random
import copy

class TransformationSpace:
    """Defines the grammar of allowed Polyhedral Transformations"""
    TRANSFORMS = [
        "UNROLL_2", "UNROLL_4", "UNROLL_8", "UNROLL_16",
        "TILE_8x8", "TILE_16x16", "TILE_32x32",
        "INTERCHANGE_0_1", "INTERCHANGE_1_2",
        "VECTORIZE_AVX2", "VECTORIZE_AVX512",
        "PARALLELIZE_OMP",
        "FUSION", "FISSION", "NO_OP"
    ]
    
    @classmethod
    def sample_random(cls, length=5):
        return [random.choice(cls.TRANSFORMS) for _ in range(length)]
        

class TransformationSearch:
    """
    Finds sequences of transformations for a target AST/program.
    Generates variations for the proxy model to learn from.
    """
    def __init__(self, target_program_id):
        self.program_id = target_program_id
        
    def random_search(self, num_samples=100, max_seq_len=10):
        """Samples uniformly from the transformation grammar"""
        sequences = []
        for _ in range(num_samples):
            length = random.randint(1, max_seq_len)
            sequences.append(TransformationSpace.sample_random(length))
        return sequences
        
    def evolutionary_search(self, initial_population=20, generations=10, mutation_rate=0.2):
        """
        Uses standard GA crossover/mutation to build variants. 
        In real context, this ties to the hardware executor proxy to establish fitness.
        """
        population = self.random_search(num_samples=initial_population)
        all_explored = copy.deepcopy(population)
        
        for gen in range(generations):
            next_gen = []
            
            # Simulated Crossover: take half from parent A, half from parent B
            for i in range(initial_population):
                p1 = random.choice(population)
                p2 = random.choice(population)
                
                mid_1 = len(p1) // 2
                mid_2 = len(p2) // 2
                
                child = p1[:mid_1] + p2[mid_2:]
                
                # Mutate: swap an element for a random transform
                if random.random() < mutation_rate and len(child) > 0:
                    idx = random.randint(0, len(child)-1)
                    child[idx] = random.choice(TransformationSpace.TRANSFORMS)
                    
                next_gen.append(child)
                all_explored.append(child)
                
            population = next_gen
            
        return all_explored
        
    def beam_search(self, proxy_model, beam_width=5, depth=5):
        """
        Model-guided greedy beam search over the transformation grammar.

        Args:
            proxy_model: Any object with a ``predict_speedup(ir_json, transform_json)``
                         method — typically a ``TiramisuHook`` instance.
            beam_width:  Number of candidates kept at each depth step.
            depth:       Maximum sequence length to explore.

        Returns:
            List of top-K sequences (each a list of transform strings),
            sorted by descending predicted speedup.
        """
        import json

        TRANSFORM_VOCAB = TransformationSpace.TRANSFORMS

        # Legality: transforms that can only appear once
        SINGLETON = {"VECTORIZE_AVX2", "VECTORIZE_AVX512", "PARALLELIZE_OMP"}

        def _applicable(seq):
            used = set(seq)
            return [t for t in TRANSFORM_VOCAB if not (t in SINGLETON and t in used)]

        # Start with the empty sequence — predicted speedup = 1.0 (no-op baseline)
        beam = [([], 1.0)]          # list of (sequence, predicted_speedup)

        for step in range(depth):
            candidates = []

            for seq, _ in beam:
                applicable = _applicable(seq)

                for t in applicable:
                    candidate_seq = seq + [t]

                    # ── REAL MODEL CALL (was previously random.uniform) ──────
                    try:
                        ir_payload        = json.dumps({"llvm_ir": getattr(proxy_model, "_last_ir", "")})
                        transform_payload = json.dumps({"transforms": [{"type": t.lower()} for t in candidate_seq]})
                        score = proxy_model.predict_speedup(ir_payload, transform_payload)
                    except Exception as e:
                        # Graceful fallback — log and skip this candidate
                        score = 1.0

                    candidates.append((candidate_seq, score))

            # Keep top beam_width by predicted speedup (descending)
            candidates.sort(key=lambda x: x[1], reverse=True)
            beam = candidates[:beam_width]

            # Early stopping: no candidate improves beyond current best
            if step > 0:
                best_score = beam[0][1] if beam else 1.0
                if best_score <= 1.001:   # essentially no prediction above baseline
                    break

        return [seq for seq, _ in beam]

if __name__ == "__main__":
    searcher = TransformationSearch("synth_0")
    print("Random Samples:", searcher.random_search(num_samples=2, max_seq_len=4))
    print("Evolutionary Generations Size:", len(searcher.evolutionary_search(generations=2)))
    print("Beam Search Top Path:", searcher.beam_search(None)[0])
