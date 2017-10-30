from Network.Net import Network

n = Network()
n.initialize('topology_01')
n.train(10000)

n1 = Network()
n1.initialize('topology_02')
n1.train(10000)

n2 = Network()
n2.initialize('topology_03')
n2.train(10000)

n3 = Network()
n3.initialize('topology_04')
n3.train(10000)

n4 = Network()
n4.initialize('topology_05')
n4.train(10000)

n5 = Network()
n5.evaluate('topology_01')
n5.evaluate('topology_02')
n5.evaluate('topology_03')
n5.evaluate('topology_04')
n5.evaluate('topology_05')
