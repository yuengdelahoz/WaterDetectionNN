from Network.Net import Network

n = Network()
n.initialize('topology_01')
n.train(10000)
n.evaluate()

n1 = Network()
n1.initialize('topology_02')
n1.train(10000)
n1.evaluate()

n2 = Network()
n2.initialize('topology_03')
n2.train(10000)
n2.evaluate()

n3 = Network()
n3.initialize('topology_04')
n3.train(10000)
n3.evaluate()

n4 = Network()
n4.initialize('topology_05')
n4.train(10000)
n4.evaluate()

