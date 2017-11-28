from Network.Net import Network

n3 = Network()
n3.initialize('topology_01')
n3.train()
n3.evaluate()

n1 = Network()
n1.initialize('topology_02')
n1.train()
n1.evaluate()


n2 = Network()
n2.initialize('topology_03')
n2.train()
n2.evaluate()

n4 = Network()
n4.initialize('topology_04')
n4.train()
n4.evaluate()

n = Network()
n.initialize('topology_05')
n.train()
# n.evaluate()
