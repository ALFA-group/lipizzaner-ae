import unittest

import torch

from aes_lipi.datasets.data_loader import create_batches
from aes_lipi.lipi_ae import initialize_nodes, get_neighbors

class TestLipiAE(unittest.TestCase):
    
    def test_get_neighbors(self):
        environment = "AutoencoderBinaryClustering"
        dataset_name = "binary_clustering_10_100"
        learning_rate = 0.01
        population_size = 3
        ann_path = ""
        batch_size = 2
        device = "cpu"
        
        training_data, _, width, height = create_batches(batch_size, dataset_name)
        nodes = initialize_nodes(learning_rate, population_size, environment, ann_path, training_data, width, height)
        print(nodes.keys())
        
        nodes[0] = get_neighbors(nodes, nodes[0], 1)
        nodes[1] = get_neighbors(nodes, nodes[1], 1)
        for e_0, e_1 in zip(nodes[0].encoders, nodes[1].encoders):
            print(id(e_0), id(e_1)) 
       
        ae_0 = nodes[0].Autoencoder(nodes[0].encoders[0], nodes[0].decoders[0])
        ae_1 = nodes[1].Autoencoder(nodes[1].encoders[-1], nodes[1].decoders[-1])
        optimizer = torch.optim.Adam(ae_0.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        for i, (batch, _) in enumerate(training_data):
            batch = batch.to(device)        
            optimizer.zero_grad()        
            loss = ae_0.compute_loss_together(batch.view(-1, ae_1.x_dim))
            loss.backward()            
            optimizer.step()
            break
        
        for name, param in nodes[0].encoders[0].named_parameters():
            print(name)
            p_1 = nodes[1].encoders[-1].get_parameter(name)
            self.assertFalse(torch.all(param == p_1))
            #for param_v in param:
            #    print(param_v)