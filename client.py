#!/usr/bin/env python
from client_stub import ClientStub


class CarEvaluationService:
  def __init__(self, addr=('127.0.0.1', 1992), serv=''):
    self.predict_stub = ClientStub('str predict(str buying, str maint, str doors, str persons, str lug_boot, str safety)', addr, serv)

  def predict(self, buying, maint, doors, persons, lug_boot, safety):
    return self.predict_stub.call({'buying': buying, 'maint': maint, 'doors': doors, 'persons': persons, 'lug_boot': lug_boot, 'safety': safety})


service = CarEvaluationService(('192.168.43.36', 1992), '')
print('Memkart inventory demo:\n')
buying = input('buying price (vhigh, high, med, low):')
maint = input('price of the maintenance (vhigh, high, med, low):')
doors = input('number of doors (2, 3, 4, 5more):')
persons = input('capacity in terms of persons to carry (2, 4, more):')
lug_boot = input('the size of luggage boot (small, med, big):')
safety = input('estimated safety of the car (low, med, high):')
accept = service.predict(buying, maint, doors, persons, lug_boot, safety)
print('result:', accept)
print('Thanks for participating.')
