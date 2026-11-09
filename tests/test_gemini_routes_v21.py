import unittest
from core.gemini_visual_v21 import routes_from_secrets, analyze_with_routes

class RoutesTest(unittest.TestCase):
    def test_array_routes(self):
        r=routes_from_secrets({'ECG_API_ROUTES':[{'name':'primary','provider':'Google Gemini','model':'gemini-test','api_key':'secret-a'}, {'name':'backup','provider':'Google Gemini','model':'gemini-other','api_key':'secret-b'}]})
        self.assertEqual(len(r),2)
    def test_single_fallback(self):
        r=routes_from_secrets({'GEMINI_API_KEY':'value'})
        self.assertEqual(len(r),1)
    def test_unsupported_provider_ignored(self):
        self.assertEqual(routes_from_secrets({'ECG_API_ROUTES':[{'provider':'Other','model':'a','api_key':'x'}]}),[])
    def test_failure_then_backup(self):
        calls=[]
        def fake(key,model,prompt,images,temp):
            calls.append(model)
            if len(calls)==1: raise RuntimeError('temporary')
            return '{"ok":true}'
        result,label=analyze_with_routes([{'model':'gemini-a','api_key':'k1'}, {'model':'gemini-b','api_key':'k2'}], 'p',[('x',b'x')],call=fake)
        self.assertEqual(calls,['gemini-a','gemini-b'])
        self.assertIn('Route 2',label)
    def test_no_leak_of_secret(self):
        def fake(*args): raise RuntimeError('a secret key was x-secret')
        with self.assertRaises(RuntimeError) as ctx:
            analyze_with_routes([{'model':'gemini-a','api_key':'x-secret'}], 'p',[('x',b'x')],call=fake)
        self.assertNotIn('x-secret',str(ctx.exception))

if __name__=='__main__': unittest.main()
