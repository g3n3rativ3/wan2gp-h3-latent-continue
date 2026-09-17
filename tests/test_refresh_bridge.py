"""Notification adapter contracts; no model weights required."""
import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location('refresh_bridge', Path(__file__).resolve().parents[1] / 'refresh_bridge.py')
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)

class Component:
    def __init__(self, ident): self._id = ident

class RefreshTests(unittest.TestCase):
    def bind(self, fn, multiple=False):
        data, signal, other = Component(1), Component(2), Component(3)
        event = SimpleNamespace(fn=fn, outputs=[other, data] if multiple else [data])
        context = SimpleNamespace(fns={0:event})
        self.assertEqual(bridge.bind_refresh_events(context, data, signal), 1)
        self.assertEqual(bridge.bind_refresh_events(context, data, signal), 0)
        self.assertIs(event.outputs[-1], signal)
        return event.fn, data, signal

    def test_cycle_identity(self):
        data={}; data['self']=data
        fn,_,_=self.bind(lambda: data)
        a,b=fn(),fn()
        self.assertIs(a[0],data)
        self.assertNotEqual(a[-1],b[-1])

    def test_multi_and_skip(self):
        fn,_,_=self.bind(lambda: (3,{}),True)
        self.assertEqual(fn()[:2],[3,{}])
        fn,_,_=self.bind(lambda: {'__type__':'update'},True)
        self.assertEqual(fn()[:2],[{'__type__':'update'}]*2)

    def test_component_dict(self):
        result={}
        fn,data,signal=self.bind(lambda: result)
        result[data]={'payload':1}
        self.assertEqual(fn()[data],{'payload':1})
        self.assertIn(signal,fn())

    def test_async(self):
        async def original(): return {'v':1}
        fn,_,_=self.bind(original)
        self.assertEqual(asyncio.run(fn())[0],{'v':1})

    def test_generators(self):
        def original():
            yield {'v':1}
            yield {'v':2}
        fn,_,_=self.bind(original)
        self.assertEqual([x[0]['v'] for x in fn()],[1,2])
        async def agen():
            yield {'v':3}
        fn,_,_=self.bind(agen)
        async def run(): return [x async for x in fn()]
        self.assertEqual(asyncio.run(run())[0][0],{'v':3})

    def test_other_forms_and_exclusion(self):
        a,b=Component(1),Component(2)
        fn=lambda: {}
        event=SimpleNamespace(fn=fn,outputs=[a])
        context=SimpleNamespace(fns={0:event})
        self.assertEqual(bridge.bind_refresh_events(context,b,a),0)
        self.assertEqual(bridge.bind_refresh_events(context,a,b,exclude=(fn,)),0)
        self.assertIs(event.fn,fn)

if __name__=='__main__': unittest.main()
