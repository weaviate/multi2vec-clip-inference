import clip
import numpy as np
from io import BytesIO
import base64
import asyncio

c = clip.Clip( False, None )

n = np.load( "testdata_preprocessed/train_1/train_1a/train_1_a_1.npz" )[ "arr_0" ]

b = BytesIO()
np.save( b, n )
b.seek( 0 )

s = base64.b64encode( b.read() )

inp = clip.ClipInput()
inp.texts = [ "" ]
inp.images = [ s ]

out = asyncio.run( c.vectorize( inp ) )
print( out )
