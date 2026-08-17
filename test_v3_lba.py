# test_v3_lba.py
# TO RUN:  python3 test_v3_lba.py -v
import unittest, torch
from atom3d.datasets import LMDBDataset
from gvp.atom3d import V3LBATransform
from torch_geometric.loader import DataLoader
from scipy.spatial.transform import Rotation
from gvp.atom3d import V3LBAModel



LMDB_VAL  = "atom3d-data/LBA/splits/split-by-sequence-identity-30/data/val"
CACHE_VAL = "/net/galaxy/home/koes/skothare/Estats/dl_model/evaluate_proteinshake_V2/v3_lba_precomputed_cache/charge_zero/val"
"""
Note: the v3 embeddings are pocket-only because the graph is pocket-only because the LMDB task is pocket-only
"""

# --- open the LMDB ONCE for the whole module ---
# LMDB refuses to open the same environment twice in one process, and multiple
# TestCase classes below need this dataset. So we open it a single time here and
# share the one instance, rather than each class opening its own.
_DS = LMDBDataset(LMDB_VAL)


class V3LBATransformTest(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.ds = _DS
        cls.transform = V3LBATransform(v3_cache_dir=CACHE_VAL)
        
        
    def test_v3emb_present_and_shaped(self):
        d = self.transform(self.ds[0])
        self.assertTrue(hasattr(d, "v3_emb")) # check if the v3_emb attribute is present
        self.assertEqual(d.v3_emb.shape[0], d.num_nodes) # check if the number of nodes is the same as the number of V3 embeddings
        self.assertEqual(d.v3_emb.shape[1], 128) # check if the number of features is 128
        
    def test_ligand_rows_zero(self):
        d = self.transform(self.ds[0]) # transform the first element of the dataset into a graph object
        self.assertTrue(torch.all(d.v3_emb[d.lig_flag] == 0)) # check if the V3 embeddings are zero for the ligand nodes
        
    def test_pocket_rows_match_cache_exactly(self):
        elem = self.ds[0]   # get the first element of the dataset
        d = self.transform(elem) # transform the first element of the dataset into a graph object
        cache = torch.load(f"{CACHE_VAL}/{elem['id']}_pocket.pt", weights_only=True) # load the V3 embeddings for the pocket nodes from the cache
        self.assertTrue(torch.equal(d.v3_emb[~d.lig_flag], cache)) # check if the V3 embeddings for the pocket nodes match the cache exactly

    def test_alignment_across_several(self):
        """Test the alignment across several elements of the dataset. The alignment is checked by comparing the number of pocket nodes in the graph with the number of V3 embeddings for the pocket nodes in the cache."""
        for i in range(5):# test the alignment across 5 elements of the dataset
            elem = self.ds[i] # get the i-th element of the dataset
            d = self.transform(elem)   # transform the i-th element of the dataset into a graph object
            cache = torch.load(f"{CACHE_VAL}/{elem['id']}_pocket.pt", weights_only=True) # load the V3 embeddings for the pocket nodes from the cache
            self.assertEqual(cache.shape[0], int((~d.lig_flag).sum())) # check if the number of pocket nodes matches the number of V3 embeddings for the pocket nodes

    def test_batching_preserves_alignment(self):
        """
        After PyG collation, v3_embd rows still line up with atoms rows.
        Checks: Does stacking graphs into a batch keep every atom aligned with its V3 embedding/vector?
        """
        n_graphs = 5
        graphs = [self.transform(self.ds[i]) for i in range(n_graphs)] # create 5 graph objects for the 5 elements of the dataset
        loader = DataLoader(graphs, batch_size=n_graphs, shuffle=False) # create a data loader with a batch size of 4; shuffle=False means no shuffling of the data and offset math is predictable since they are in order.

        """
        The DataLoaderconstructor takes the list of graphs and a batch size, and returns an object that, when iterated, hands us collated batches. For graphs specifically, "collate" means something special.

        Why create a batch in the test? Because training will use batches, and batching is where alignment could silently break. This is exactly what needs to be tested here, The test isn't testing the DataLoader but  it's testing that the v3_emb survives the thing training does to it. We deliberately reproduce the batching step to check it doesn't scramble the embeddings.
        """

        batch = next(iter(loader)) # iter(loader) returns an iterator over the batches in the data loader; pulls the first batch from the data loader; batch size is 4 as passed in the DataLoader constructor

        #now, total node count must match across the node-aligned attributes
        # after concatenation, v3_emb, atoms, and x must all have same total number of rows (total nodes across 4 graphs)
        self.assertEqual(batch.v3_emb.shape[0], batch.atoms.shape[0]) # total number of nodes must match across the node-aligned attributes (atoms)
        self.assertEqual(batch.v3_emb.shape[0], batch.x.shape[0]) # total number of nodes must match the x feature matrix
        """
        Note, after calling self.transform(), the V3LBATransform class attaches the V3 embeddings to the graph object as a node-aligned attribute called v3_emb.
        """

        # per=graph pocket rows myst equal each protein's own cache in order
        """
        The deep check. Since PyG concatenates, graph 0 occupies rows 0…n0, graph 1 rows n0…n0+n1, etc. offset tracks where each graph starts. For each graph: rebuild it (g), get its node count (n), slice out its block of the batch (block = batch.v3_emb[offset:offset+n]), then check that block's pocket rows equal that protein's cache. offset += n advances to the next graph's block.
        """
        offset = 0
        for i in range(n_graphs): # iterate over the 5 elements of the dataset
            g = graphs[i] # get the i-th graph
            n = g.num_nodes # get the number of nodes in the i-th graph
            cache = torch.load(f"{CACHE_VAL}/{self.ds[i]['id']}_pocket.pt", weights_only=True) # load the V3 embeddings for the pocket nodes from the cache
            block = batch.v3_emb[offset:offset+n] # get the V3 embeddings for the pocket nodes for the i-th graph
            self.assertEqual(block.shape[0], n) # guard: block is non-empty & right size
            self.assertTrue(torch.equal(block[~g.lig_flag], cache)) # check if the V3 embeddings for the pocket nodes for the i-th graph match the cache exactly
            offset += n # increment the offset by the number of nodes in the i-th graph

    def test_full_row_binding(self):
        """Every pocket node's 128-vector belongs to THAT node —shuffle-proof check.
        
        Checks: Does every pocket node's 128-vector belong to THAT node? i.e. is each atom holding its own 128-vector?
        """
        elem = self.ds[0] # get the first element of the dataset
        d = self.transform(elem) # transform the first element of the dataset into a graph object
        cache = torch.load(f"{CACHE_VAL}/{elem['id']}_pocket.pt", weights_only=True) # load the V3 embeddings for the pocket nodes from the cache
        pocket_rows = d.v3_emb[~d.lig_flag] # get the V3 embeddings for the pocket nodes
        for j in range(0, cache.shape[0], 25):# spot-check every 25th atom arbitrarily
            self.assertTrue(torch.equal(pocket_rows[j], cache[j])) # main question: does every pocket node's 128-vector belong to THAT node?


# Test the V3LBAModel--------------------------------------
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_fake_batch(n_nodes=40, n_edges=200, n_graphs=3):
    """Fabricate a minimal batch with every attribute V3LBAModel.forward reads.
    No cache, no LMDB — just tensors of the right shapes/dtypes."""
    class B: pass
    b = B()
    b.atoms      = torch.randint(0, 9, (n_nodes,), device=device)          # element ids 0-8
    b.v3_emb     = torch.randn(n_nodes, 128, device=device)                # fake V3 embedding
    b.edge_index = torch.randint(0, n_nodes, (2, n_edges), device=device)  # random edges
    b.edge_s     = torch.randn(n_edges, 16, device=device)                 # edge scalars (num_rbf=16)
    b.edge_v     = torch.randn(n_edges, 1, 3, device=device)               # edge vectors
    b.batch      = torch.randint(0, n_graphs, (n_nodes,), device=device)   # node -> graph id
    b.x          = torch.randn(n_nodes, 3, device=device)                  # coords (unused by forward, present for realism)
    return b, n_graphs


class V3LBAModelTest(unittest.TestCase):

    def test_forward_runs_and_output_shape(self):
        """Model ingests 137 scalars end-to-end and emits one scalar per graph."""
        model = V3LBAModel().to(device).eval()
        batch, n_graphs = make_fake_batch(n_graphs=3)
        with torch.no_grad():
            out = model(batch)
        # scatter_mean pools to one row per graph; dense squeezes to a scalar each
        self.assertEqual(out.shape, (n_graphs,))
        self.assertTrue(torch.isfinite(out).all())          # no NaNs/infs

    def test_W_v_input_width_is_137(self):
        """The rebuilt W_v must accept 137 scalars, not the baseline 9."""
        model = V3LBAModel()
        # W_v = Sequential(LayerNorm, GVP); the GVP is index 1, its scalar input
        # projection ws.in_features encodes the expected scalar input width.
        gvp_layer = model.W_v[1]
        self.assertEqual(gvp_layer.ws.in_features, 137)

    def test_W_v_input_width_matches_v3dim(self):
        model = V3LBAModel(v3_dim=128)
        self.assertEqual(model.W_v[1].ws.in_features, 9 + model.v3_dim)   # 137

    def test_rotation_invariance(self):
        """Rotating input coordinates must NOT change the predicted scalar.
        V3LBAModel returns a rotation-INVARIANT scalar (pKd). We inject 128
        scalar (rotation-invariant) features, so this property must survive.
        We rotate the only geometry the forward pass consumes: edge_v."""
        model = V3LBAModel().to(device).eval()
        batch, _ = make_fake_batch()

        R = torch.as_tensor(Rotation.random().as_matrix(),
                            dtype=torch.float32, device=device)

        with torch.no_grad():
            out_original = model(batch)
            batch.edge_v = batch.edge_v @ R                 # rotate edge direction vectors
            out_rotated  = model(batch)

        # A correct GVP scalar output is invariant to input rotation.
        self.assertTrue(torch.allclose(out_original, out_rotated, atol=1e-4, rtol=1e-3))

class V3EndToEndTest(unittest.TestCase):
    """Real cache tensors through the real model — the pre-flight before SLURM."""

    @classmethod
    def setUpClass(cls):
        cls.ds = _DS
        cls.transform = V3LBATransform(v3_cache_dir=CACHE_VAL)

    def test_real_batch_through_model(self):
        # Build a real batch from real data via the real transform + real loader.
        graphs = [self.transform(self.ds[i]) for i in range(4)]
        loader = DataLoader(graphs, batch_size=4, shuffle=False)
        batch = next(iter(loader)).to(device)

        model = V3LBAModel().to(device).eval()
        with torch.no_grad():
            out = model(batch)                      # real 128-dim cache -> real 137 W_v

        self.assertEqual(out.shape, (4,))           # one pKd per graph
        self.assertTrue(torch.isfinite(out).all())  # no NaN/inf on real inputs

if __name__ == "__main__": # runs the tests if the script is run directly
    unittest.main() 
