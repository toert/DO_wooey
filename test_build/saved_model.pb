’
«..
+
Abs
x"T
y"T"
Ttype:	
2	
9
Add
x"T
y"T
z"T"
Ttype:
2	
T
AddN
inputs"T*N
sum"T"
Nint(0"
Ttype:
2	
ė
	ApplyAdam
var"T	
m"T	
v"T
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T"
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
p
	AssignAdd
ref"T

value"T

output_ref"T"
Ttype:
2	"
use_lockingbool( 
{
BiasAdd

value"T	
bias"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
{
BiasAddGrad
out_backprop"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
I
ConcatOffset

concat_dim
shape*N
offset*N"
Nint(0
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
Č
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
ī
Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
ķ
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
:
Elu
features"T
activations"T"
Ttype:
2
K
EluGrad
	gradients"T
outputs"T
	backprops"T"
Ttype:
2
4
Fill
dims

value"T
output"T"	
Ttype
+
Floor
x"T
y"T"
Ttype:
2
>
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
7
FloorMod
x"T
y"T
z"T"
Ttype:
2	
S
HistogramSummary
tag
values"T
summary"
Ttype0:
2		
.
Identity

input"T
output"T"	
Ttype
1
L2Loss
t"T
output"T"
Ttype:
2
+
Log
x"T
y"T"
Ttype:	
2
:
Maximum
x"T
y"T
z"T"
Ttype:	
2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
8
MergeSummary
inputs*N
summary"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
<
Mul
x"T
y"T
z"T"
Ttype:
2	
-
Neg
x"T
y"T"
Ttype:
	2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
5
Pow
x"T
y"T
z"T"
Ttype:
	2	

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
}
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
`
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:
2	
)
Rank

input"T

output"	
Ttype
=
RealDiv
x"T
y"T
z"T"
Ttype:
2	
4

Reciprocal
x"T
y"T"
Ttype:
	2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
M
ScalarSummary
tags
values"T
summary"
Ttype:
2		
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
.
Sign
x"T
y"T"
Ttype:
	2	
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
8
Softmax
logits"T
softmax"T"
Ttype:
2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 

StridedSliceGrad
shape"Index
begin"Index
end"Index
strides"Index
dy"T
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
9
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring "train*1.4.02v1.4.0-rc1-11-g130a514Õ
t
XPlaceholder*
dtype0*$
shape:’’’’’’’’’2*/
_output_shapes
:’’’’’’’’’2
d
yPlaceholder*
dtype0*
shape:’’’’’’’’’*'
_output_shapes
:’’’’’’’’’
„
,cnn1/kernel/Initializer/random_uniform/shapeConst*%
valueB"            *
dtype0*
_class
loc:@cnn1/kernel*
_output_shapes
:

*cnn1/kernel/Initializer/random_uniform/minConst*
valueB
 *č!æ*
dtype0*
_class
loc:@cnn1/kernel*
_output_shapes
: 

*cnn1/kernel/Initializer/random_uniform/maxConst*
valueB
 *č!?*
dtype0*
_class
loc:@cnn1/kernel*
_output_shapes
: 
ź
4cnn1/kernel/Initializer/random_uniform/RandomUniformRandomUniform,cnn1/kernel/Initializer/random_uniform/shape*

seed *
seed2 *
dtype0*
T0*
_class
loc:@cnn1/kernel*&
_output_shapes
:
Ź
*cnn1/kernel/Initializer/random_uniform/subSub*cnn1/kernel/Initializer/random_uniform/max*cnn1/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@cnn1/kernel*
_output_shapes
: 
ä
*cnn1/kernel/Initializer/random_uniform/mulMul4cnn1/kernel/Initializer/random_uniform/RandomUniform*cnn1/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@cnn1/kernel*&
_output_shapes
:
Ö
&cnn1/kernel/Initializer/random_uniformAdd*cnn1/kernel/Initializer/random_uniform/mul*cnn1/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@cnn1/kernel*&
_output_shapes
:
Æ
cnn1/kernel
VariableV2*
shape:*
dtype0*
	container *
shared_name *
_class
loc:@cnn1/kernel*&
_output_shapes
:
Ė
cnn1/kernel/AssignAssigncnn1/kernel&cnn1/kernel/Initializer/random_uniform*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn1/kernel*&
_output_shapes
:
z
cnn1/kernel/readIdentitycnn1/kernel*
T0*
_class
loc:@cnn1/kernel*&
_output_shapes
:

,cnn1/kernel/Regularizer/l2_regularizer/scaleConst*
valueB
 *_p	0*
dtype0*
_class
loc:@cnn1/kernel*
_output_shapes
: 

-cnn1/kernel/Regularizer/l2_regularizer/L2LossL2Losscnn1/kernel/read*
T0*
_class
loc:@cnn1/kernel*
_output_shapes
: 
Ė
&cnn1/kernel/Regularizer/l2_regularizerMul,cnn1/kernel/Regularizer/l2_regularizer/scale-cnn1/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_class
loc:@cnn1/kernel*
_output_shapes
: 

cnn1/bias/Initializer/zerosConst*
valueB*    *
dtype0*
_class
loc:@cnn1/bias*
_output_shapes
:

	cnn1/bias
VariableV2*
shape:*
dtype0*
	container *
shared_name *
_class
loc:@cnn1/bias*
_output_shapes
:
®
cnn1/bias/AssignAssign	cnn1/biascnn1/bias/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn1/bias*
_output_shapes
:
h
cnn1/bias/readIdentity	cnn1/bias*
T0*
_class
loc:@cnn1/bias*
_output_shapes
:
c
cnn1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
¼
cnn1/Conv2DConv2DXcnn1/kernel/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingVALID*
data_formatNHWC*/
_output_shapes
:’’’’’’’’’0

cnn1/BiasAddBiasAddcnn1/Conv2Dcnn1/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:’’’’’’’’’0
W
cnn1/EluElucnn1/BiasAdd*
T0*/
_output_shapes
:’’’’’’’’’0
„
,cnn2/kernel/Initializer/random_uniform/shapeConst*%
valueB"   0         *
dtype0*
_class
loc:@cnn2/kernel*
_output_shapes
:

*cnn2/kernel/Initializer/random_uniform/minConst*
valueB
 *²_½*
dtype0*
_class
loc:@cnn2/kernel*
_output_shapes
: 

*cnn2/kernel/Initializer/random_uniform/maxConst*
valueB
 *²_=*
dtype0*
_class
loc:@cnn2/kernel*
_output_shapes
: 
ź
4cnn2/kernel/Initializer/random_uniform/RandomUniformRandomUniform,cnn2/kernel/Initializer/random_uniform/shape*

seed *
seed2 *
dtype0*
T0*
_class
loc:@cnn2/kernel*&
_output_shapes
:0
Ź
*cnn2/kernel/Initializer/random_uniform/subSub*cnn2/kernel/Initializer/random_uniform/max*cnn2/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@cnn2/kernel*
_output_shapes
: 
ä
*cnn2/kernel/Initializer/random_uniform/mulMul4cnn2/kernel/Initializer/random_uniform/RandomUniform*cnn2/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@cnn2/kernel*&
_output_shapes
:0
Ö
&cnn2/kernel/Initializer/random_uniformAdd*cnn2/kernel/Initializer/random_uniform/mul*cnn2/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@cnn2/kernel*&
_output_shapes
:0
Æ
cnn2/kernel
VariableV2*
shape:0*
dtype0*
	container *
shared_name *
_class
loc:@cnn2/kernel*&
_output_shapes
:0
Ė
cnn2/kernel/AssignAssigncnn2/kernel&cnn2/kernel/Initializer/random_uniform*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn2/kernel*&
_output_shapes
:0
z
cnn2/kernel/readIdentitycnn2/kernel*
T0*
_class
loc:@cnn2/kernel*&
_output_shapes
:0

,cnn2/kernel/Regularizer/l2_regularizer/scaleConst*
valueB
 *_p	0*
dtype0*
_class
loc:@cnn2/kernel*
_output_shapes
: 

-cnn2/kernel/Regularizer/l2_regularizer/L2LossL2Losscnn2/kernel/read*
T0*
_class
loc:@cnn2/kernel*
_output_shapes
: 
Ė
&cnn2/kernel/Regularizer/l2_regularizerMul,cnn2/kernel/Regularizer/l2_regularizer/scale-cnn2/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_class
loc:@cnn2/kernel*
_output_shapes
: 

cnn2/bias/Initializer/zerosConst*
valueB*    *
dtype0*
_class
loc:@cnn2/bias*
_output_shapes
:

	cnn2/bias
VariableV2*
shape:*
dtype0*
	container *
shared_name *
_class
loc:@cnn2/bias*
_output_shapes
:
®
cnn2/bias/AssignAssign	cnn2/biascnn2/bias/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn2/bias*
_output_shapes
:
h
cnn2/bias/readIdentity	cnn2/bias*
T0*
_class
loc:@cnn2/bias*
_output_shapes
:
c
cnn2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Ć
cnn2/Conv2DConv2Dcnn1/Elucnn2/kernel/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingVALID*
data_formatNHWC*/
_output_shapes
:’’’’’’’’’

cnn2/BiasAddBiasAddcnn2/Conv2Dcnn2/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:’’’’’’’’’
W
cnn2/EluElucnn2/BiasAdd*
T0*/
_output_shapes
:’’’’’’’’’
„
,cnn3/kernel/Initializer/random_uniform/shapeConst*%
valueB"            *
dtype0*
_class
loc:@cnn3/kernel*
_output_shapes
:

*cnn3/kernel/Initializer/random_uniform/minConst*
valueB
 *wÖæ*
dtype0*
_class
loc:@cnn3/kernel*
_output_shapes
: 

*cnn3/kernel/Initializer/random_uniform/maxConst*
valueB
 *wÖ?*
dtype0*
_class
loc:@cnn3/kernel*
_output_shapes
: 
ź
4cnn3/kernel/Initializer/random_uniform/RandomUniformRandomUniform,cnn3/kernel/Initializer/random_uniform/shape*

seed *
seed2 *
dtype0*
T0*
_class
loc:@cnn3/kernel*&
_output_shapes
:
Ź
*cnn3/kernel/Initializer/random_uniform/subSub*cnn3/kernel/Initializer/random_uniform/max*cnn3/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@cnn3/kernel*
_output_shapes
: 
ä
*cnn3/kernel/Initializer/random_uniform/mulMul4cnn3/kernel/Initializer/random_uniform/RandomUniform*cnn3/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@cnn3/kernel*&
_output_shapes
:
Ö
&cnn3/kernel/Initializer/random_uniformAdd*cnn3/kernel/Initializer/random_uniform/mul*cnn3/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@cnn3/kernel*&
_output_shapes
:
Æ
cnn3/kernel
VariableV2*
shape:*
dtype0*
	container *
shared_name *
_class
loc:@cnn3/kernel*&
_output_shapes
:
Ė
cnn3/kernel/AssignAssigncnn3/kernel&cnn3/kernel/Initializer/random_uniform*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn3/kernel*&
_output_shapes
:
z
cnn3/kernel/readIdentitycnn3/kernel*
T0*
_class
loc:@cnn3/kernel*&
_output_shapes
:

,cnn3/kernel/Regularizer/l2_regularizer/scaleConst*
valueB
 *_p	0*
dtype0*
_class
loc:@cnn3/kernel*
_output_shapes
: 

-cnn3/kernel/Regularizer/l2_regularizer/L2LossL2Losscnn3/kernel/read*
T0*
_class
loc:@cnn3/kernel*
_output_shapes
: 
Ė
&cnn3/kernel/Regularizer/l2_regularizerMul,cnn3/kernel/Regularizer/l2_regularizer/scale-cnn3/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_class
loc:@cnn3/kernel*
_output_shapes
: 

cnn3/bias/Initializer/zerosConst*
valueB*    *
dtype0*
_class
loc:@cnn3/bias*
_output_shapes
:

	cnn3/bias
VariableV2*
shape:*
dtype0*
	container *
shared_name *
_class
loc:@cnn3/bias*
_output_shapes
:
®
cnn3/bias/AssignAssign	cnn3/biascnn3/bias/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn3/bias*
_output_shapes
:
h
cnn3/bias/readIdentity	cnn3/bias*
T0*
_class
loc:@cnn3/bias*
_output_shapes
:
c
cnn3/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Ć
cnn3/Conv2DConv2Dcnn2/Elucnn3/kernel/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingVALID*
data_formatNHWC*/
_output_shapes
:’’’’’’’’’

cnn3/BiasAddBiasAddcnn3/Conv2Dcnn3/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:’’’’’’’’’
W
SqueezeSqueezecnn3/BiasAdd*
T0*
squeeze_dims
 *
_output_shapes
:
U
ShapeShapeSqueeze*
T0*
out_type0*#
_output_shapes
:’’’’’’’’’
6
RankRankSqueeze*
T0*
_output_shapes
: 
W
Shape_1ShapeSqueeze*
T0*
out_type0*#
_output_shapes
:’’’’’’’’’
G
Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
8
SubSubRankSub/y*
T0*
_output_shapes
: 
R
Slice/beginPackSub*
N*
T0*

axis *
_output_shapes
:
T

Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
b
SliceSliceShape_1Slice/begin
Slice/size*
T0*
Index0*
_output_shapes
:
b
concat/values_0Const*
valueB:
’’’’’’’’’*
dtype0*
_output_shapes
:
M
concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
q
concatConcatV2concat/values_0Sliceconcat/axis*
N*
T0*

Tidx0*
_output_shapes
:
l
ReshapeReshapeSqueezeconcat*
T0*
Tshape0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’
V
SoftmaxSoftmaxReshape*
T0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’
`
calculated_portfolioReshapeSoftmaxShape*
T0*
Tshape0*
_output_shapes
:
W
portfolio/tagConst*
valueB B	portfolio*
dtype0*
_output_shapes
: 
c
	portfolioHistogramSummaryportfolio/tagcalculated_portfolio*
T0*
_output_shapes
: 
_
cnn1/kernel_1/tagConst*
valueB Bcnn1/kernel_1*
dtype0*
_output_shapes
: 
x
cnn1/kernel_1/valuesPackcnn1/kernel/read*
N*
T0*

axis **
_output_shapes
:
k
cnn1/kernel_1HistogramSummarycnn1/kernel_1/tagcnn1/kernel_1/values*
T0*
_output_shapes
: 
[
cnn1/bias_1/tagConst*
valueB Bcnn1/bias_1*
dtype0*
_output_shapes
: 
h
cnn1/bias_1/valuesPackcnn1/bias/read*
N*
T0*

axis *
_output_shapes

:
e
cnn1/bias_1HistogramSummarycnn1/bias_1/tagcnn1/bias_1/values*
T0*
_output_shapes
: 
_
cnn2/kernel_1/tagConst*
valueB Bcnn2/kernel_1*
dtype0*
_output_shapes
: 
x
cnn2/kernel_1/valuesPackcnn2/kernel/read*
N*
T0*

axis **
_output_shapes
:0
k
cnn2/kernel_1HistogramSummarycnn2/kernel_1/tagcnn2/kernel_1/values*
T0*
_output_shapes
: 
[
cnn2/bias_1/tagConst*
valueB Bcnn2/bias_1*
dtype0*
_output_shapes
: 
h
cnn2/bias_1/valuesPackcnn2/bias/read*
N*
T0*

axis *
_output_shapes

:
e
cnn2/bias_1HistogramSummarycnn2/bias_1/tagcnn2/bias_1/values*
T0*
_output_shapes
: 
_
cnn3/kernel_1/tagConst*
valueB Bcnn3/kernel_1*
dtype0*
_output_shapes
: 
x
cnn3/kernel_1/valuesPackcnn3/kernel/read*
N*
T0*

axis **
_output_shapes
:
k
cnn3/kernel_1HistogramSummarycnn3/kernel_1/tagcnn3/kernel_1/values*
T0*
_output_shapes
: 
[
cnn3/bias_1/tagConst*
valueB Bcnn3/bias_1*
dtype0*
_output_shapes
: 
h
cnn3/bias_1/valuesPackcnn3/bias/read*
N*
T0*

axis *
_output_shapes

:
e
cnn3/bias_1HistogramSummarycnn3/bias_1/tagcnn3/bias_1/values*
T0*
_output_shapes
: 
H
feeConst*
valueB
 *o;*
dtype0*
_output_shapes
: 
y
buffer_valueConst*5
value,B*"  ?                        *
dtype0*
_output_shapes

:
]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
h
strided_slice/stack_1Const*
valueB:
’’’’’’’’’*
dtype0*
_output_shapes
:
_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

strided_sliceStridedSlicecalculated_portfoliostrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
T0*
Index0*

begin_mask *
end_mask *
ellipsis_mask *
new_axis_mask *
shrink_axis_mask *
_output_shapes
:
^
previous_step_portfolio/axisConst*
value	B : *
dtype0*
_output_shapes
: 
„
previous_step_portfolioConcatV2buffer_valuestrided_sliceprevious_step_portfolio/axis*
N*
T0*

Tidx0*'
_output_shapes
:’’’’’’’’’
X
pn1Mulyprevious_step_portfolio*
T0*'
_output_shapes
:’’’’’’’’’
W
pn2/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
u
pn2Sumpn1pn2/reduction_indices*
	keep_dims(*
T0*

Tidx0*'
_output_shapes
:’’’’’’’’’
X
current_portfolioRealDivpn1pn2*
T0*'
_output_shapes
:’’’’’’’’’
f
strided_slice_1/stackConst*
valueB"       *
dtype0*
_output_shapes
:
h
strided_slice_1/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:
h
strided_slice_1/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:

strided_slice_1StridedSlicecurrent_portfoliostrided_slice_1/stackstrided_slice_1/stack_1strided_slice_1/stack_2*
T0*
Index0*

begin_mask*
end_mask*
ellipsis_mask *
new_axis_mask *
shrink_axis_mask *'
_output_shapes
:’’’’’’’’’
f
strided_slice_2/stackConst*
valueB"       *
dtype0*
_output_shapes
:
h
strided_slice_2/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:
h
strided_slice_2/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:

strided_slice_2StridedSlicecalculated_portfoliostrided_slice_2/stackstrided_slice_2/stack_1strided_slice_2/stack_2*
T0*
Index0*

begin_mask*
end_mask*
ellipsis_mask *
new_axis_mask *
shrink_axis_mask *
_output_shapes
:
Q
part1Substrided_slice_1strided_slice_2*
T0*
_output_shapes
:
4
AbsAbspart1*
T0*
_output_shapes
:
Y
part2/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
j
part2SumAbspart2/reduction_indices*
	keep_dims(*
T0*

Tidx0*
_output_shapes
:
J
ConstConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
9
MulMulfeepart2*
T0*
_output_shapes
:
C
shrink_vectorSubConstMul*
T0*
_output_shapes
:
@
a3Mulshrink_vectorpn2*
T0*
_output_shapes
:
0
a4Loga3*
T0*
_output_shapes
:
3
Rank_1Ranka4*
T0*
_output_shapes
: 
M
range/startConst*
value	B : *
dtype0*
_output_shapes
: 
M
range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
a
rangeRangerange/startRank_1range/delta*

Tidx0*#
_output_shapes
:’’’’’’’’’
Y
rewardMeana4range*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
»
	full_lossAddNreward&cnn1/kernel/Regularizer/l2_regularizer&cnn2/kernel/Regularizer/l2_regularizer&cnn3/kernel/Regularizer/l2_regularizer*
N*
T0*
_output_shapes
: 
I
mu/tagConst*
value
B Bmu*
dtype0*
_output_shapes
: 
N
muHistogramSummarymu/tagshrink_vector*
T0*
_output_shapes
: 
X
base_loss/tagsConst*
valueB B	base_loss*
dtype0*
_output_shapes
: 
S
	base_lossScalarSummarybase_loss/tagsreward*
T0*
_output_shapes
: 
\
full_loss_1/tagsConst*
valueB Bfull_loss_1*
dtype0*
_output_shapes
: 
Z
full_loss_1ScalarSummaryfull_loss_1/tags	full_loss*
T0*
_output_shapes
: 
W
Sum/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 

SumSumcurrent_portfolioSum/reduction_indices*
	keep_dims( *
T0*

Tidx0*#
_output_shapes
:’’’’’’’’’
O
	sum_p/tagConst*
valueB Bsum_p*
dtype0*
_output_shapes
: 
J
sum_pHistogramSummary	sum_p/tagSum*
T0*
_output_shapes
: 
X
Variable/initial_valueConst*
value	B : *
dtype0*
_output_shapes
: 
l
Variable
VariableV2*
shape: *
dtype0*
	container *
shared_name *
_output_shapes
: 
¢
Variable/AssignAssignVariableVariable/initial_value*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable*
_output_shapes
: 
a
Variable/readIdentityVariable*
T0*
_class
loc:@Variable*
_output_shapes
: 
c
ExponentialDecay/learning_rateConst*
valueB
 *o;*
dtype0*
_output_shapes
: 
\
ExponentialDecay/CastCastVariable/read*

SrcT0*

DstT0*
_output_shapes
: 
\
ExponentialDecay/Cast_1/xConst*
value
B :ø*
dtype0*
_output_shapes
: 
j
ExponentialDecay/Cast_1CastExponentialDecay/Cast_1/x*

SrcT0*

DstT0*
_output_shapes
: 
^
ExponentialDecay/Cast_2/xConst*
valueB
 *Āu?*
dtype0*
_output_shapes
: 
t
ExponentialDecay/truedivRealDivExponentialDecay/CastExponentialDecay/Cast_1*
T0*
_output_shapes
: 
Z
ExponentialDecay/FloorFloorExponentialDecay/truediv*
T0*
_output_shapes
: 
o
ExponentialDecay/PowPowExponentialDecay/Cast_2/xExponentialDecay/Floor*
T0*
_output_shapes
: 
n
ExponentialDecayMulExponentialDecay/learning_rateExponentialDecay/Pow*
T0*
_output_shapes
: 
6
NegNeg	full_loss*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
T
gradients/ConstConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Y
gradients/FillFillgradients/Shapegradients/Const*
T0*
_output_shapes
: 
N
gradients/Neg_grad/NegNeggradients/Fill*
T0*
_output_shapes
: 
J
)gradients/full_loss_grad/tuple/group_depsNoOp^gradients/Neg_grad/Neg
Ķ
1gradients/full_loss_grad/tuple/control_dependencyIdentitygradients/Neg_grad/Neg*^gradients/full_loss_grad/tuple/group_deps*
T0*)
_class
loc:@gradients/Neg_grad/Neg*
_output_shapes
: 
Ļ
3gradients/full_loss_grad/tuple/control_dependency_1Identitygradients/Neg_grad/Neg*^gradients/full_loss_grad/tuple/group_deps*
T0*)
_class
loc:@gradients/Neg_grad/Neg*
_output_shapes
: 
Ļ
3gradients/full_loss_grad/tuple/control_dependency_2Identitygradients/Neg_grad/Neg*^gradients/full_loss_grad/tuple/group_deps*
T0*)
_class
loc:@gradients/Neg_grad/Neg*
_output_shapes
: 
Ļ
3gradients/full_loss_grad/tuple/control_dependency_3Identitygradients/Neg_grad/Neg*^gradients/full_loss_grad/tuple/group_deps*
T0*)
_class
loc:@gradients/Neg_grad/Neg*
_output_shapes
: 
f
gradients/reward_grad/ShapeShapea4*
T0*
out_type0*#
_output_shapes
:’’’’’’’’’
 
gradients/reward_grad/SizeSizegradients/reward_grad/Shape*
T0*
out_type0*.
_class$
" loc:@gradients/reward_grad/Shape*
_output_shapes
: 
”
gradients/reward_grad/addAddrangegradients/reward_grad/Size*
T0*.
_class$
" loc:@gradients/reward_grad/Shape*#
_output_shapes
:’’’’’’’’’
ŗ
gradients/reward_grad/modFloorModgradients/reward_grad/addgradients/reward_grad/Size*
T0*.
_class$
" loc:@gradients/reward_grad/Shape*#
_output_shapes
:’’’’’’’’’
¦
gradients/reward_grad/Shape_1Shapegradients/reward_grad/mod*
T0*
out_type0*.
_class$
" loc:@gradients/reward_grad/Shape*
_output_shapes
:

!gradients/reward_grad/range/startConst*
value	B : *
dtype0*.
_class$
" loc:@gradients/reward_grad/Shape*
_output_shapes
: 

!gradients/reward_grad/range/deltaConst*
value	B :*
dtype0*.
_class$
" loc:@gradients/reward_grad/Shape*
_output_shapes
: 
ē
gradients/reward_grad/rangeRange!gradients/reward_grad/range/startgradients/reward_grad/Size!gradients/reward_grad/range/delta*

Tidx0*.
_class$
" loc:@gradients/reward_grad/Shape*#
_output_shapes
:’’’’’’’’’

 gradients/reward_grad/Fill/valueConst*
value	B :*
dtype0*.
_class$
" loc:@gradients/reward_grad/Shape*
_output_shapes
: 
Į
gradients/reward_grad/FillFillgradients/reward_grad/Shape_1 gradients/reward_grad/Fill/value*
T0*.
_class$
" loc:@gradients/reward_grad/Shape*#
_output_shapes
:’’’’’’’’’

#gradients/reward_grad/DynamicStitchDynamicStitchgradients/reward_grad/rangegradients/reward_grad/modgradients/reward_grad/Shapegradients/reward_grad/Fill*
N*
T0*.
_class$
" loc:@gradients/reward_grad/Shape*#
_output_shapes
:’’’’’’’’’

gradients/reward_grad/Maximum/yConst*
value	B :*
dtype0*.
_class$
" loc:@gradients/reward_grad/Shape*
_output_shapes
: 
Ģ
gradients/reward_grad/MaximumMaximum#gradients/reward_grad/DynamicStitchgradients/reward_grad/Maximum/y*
T0*.
_class$
" loc:@gradients/reward_grad/Shape*#
_output_shapes
:’’’’’’’’’
Ä
gradients/reward_grad/floordivFloorDivgradients/reward_grad/Shapegradients/reward_grad/Maximum*
T0*.
_class$
" loc:@gradients/reward_grad/Shape*#
_output_shapes
:’’’’’’’’’
±
gradients/reward_grad/ReshapeReshape1gradients/full_loss_grad/tuple/control_dependency#gradients/reward_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:

gradients/reward_grad/TileTilegradients/reward_grad/Reshapegradients/reward_grad/floordiv*
T0*

Tmultiples0*
_output_shapes
:
h
gradients/reward_grad/Shape_2Shapea4*
T0*
out_type0*#
_output_shapes
:’’’’’’’’’
l
gradients/reward_grad/Shape_3Shapereward*
T0*
out_type0*#
_output_shapes
:’’’’’’’’’

gradients/reward_grad/ConstConst*
valueB: *
dtype0*0
_class&
$"loc:@gradients/reward_grad/Shape_2*
_output_shapes
:
Ī
gradients/reward_grad/ProdProdgradients/reward_grad/Shape_2gradients/reward_grad/Const*
	keep_dims( *
T0*

Tidx0*0
_class&
$"loc:@gradients/reward_grad/Shape_2*
_output_shapes
: 

gradients/reward_grad/Const_1Const*
valueB: *
dtype0*0
_class&
$"loc:@gradients/reward_grad/Shape_2*
_output_shapes
:
Ņ
gradients/reward_grad/Prod_1Prodgradients/reward_grad/Shape_3gradients/reward_grad/Const_1*
	keep_dims( *
T0*

Tidx0*0
_class&
$"loc:@gradients/reward_grad/Shape_2*
_output_shapes
: 

!gradients/reward_grad/Maximum_1/yConst*
value	B :*
dtype0*0
_class&
$"loc:@gradients/reward_grad/Shape_2*
_output_shapes
: 
¾
gradients/reward_grad/Maximum_1Maximumgradients/reward_grad/Prod_1!gradients/reward_grad/Maximum_1/y*
T0*0
_class&
$"loc:@gradients/reward_grad/Shape_2*
_output_shapes
: 
¼
 gradients/reward_grad/floordiv_1FloorDivgradients/reward_grad/Prodgradients/reward_grad/Maximum_1*
T0*0
_class&
$"loc:@gradients/reward_grad/Shape_2*
_output_shapes
: 
t
gradients/reward_grad/CastCast gradients/reward_grad/floordiv_1*

SrcT0*

DstT0*
_output_shapes
: 

gradients/reward_grad/truedivRealDivgradients/reward_grad/Tilegradients/reward_grad/Cast*
T0*
_output_shapes
:
~
;gradients/cnn1/kernel/Regularizer/l2_regularizer_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 

=gradients/cnn1/kernel/Regularizer/l2_regularizer_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 

Kgradients/cnn1/kernel/Regularizer/l2_regularizer_grad/BroadcastGradientArgsBroadcastGradientArgs;gradients/cnn1/kernel/Regularizer/l2_regularizer_grad/Shape=gradients/cnn1/kernel/Regularizer/l2_regularizer_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
Å
9gradients/cnn1/kernel/Regularizer/l2_regularizer_grad/mulMul3gradients/full_loss_grad/tuple/control_dependency_1-cnn1/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 

9gradients/cnn1/kernel/Regularizer/l2_regularizer_grad/SumSum9gradients/cnn1/kernel/Regularizer/l2_regularizer_grad/mulKgradients/cnn1/kernel/Regularizer/l2_regularizer_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
ļ
=gradients/cnn1/kernel/Regularizer/l2_regularizer_grad/ReshapeReshape9gradients/cnn1/kernel/Regularizer/l2_regularizer_grad/Sum;gradients/cnn1/kernel/Regularizer/l2_regularizer_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
Ę
;gradients/cnn1/kernel/Regularizer/l2_regularizer_grad/mul_1Mul,cnn1/kernel/Regularizer/l2_regularizer/scale3gradients/full_loss_grad/tuple/control_dependency_1*
T0*
_output_shapes
: 

;gradients/cnn1/kernel/Regularizer/l2_regularizer_grad/Sum_1Sum;gradients/cnn1/kernel/Regularizer/l2_regularizer_grad/mul_1Mgradients/cnn1/kernel/Regularizer/l2_regularizer_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
õ
?gradients/cnn1/kernel/Regularizer/l2_regularizer_grad/Reshape_1Reshape;gradients/cnn1/kernel/Regularizer/l2_regularizer_grad/Sum_1=gradients/cnn1/kernel/Regularizer/l2_regularizer_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Š
Fgradients/cnn1/kernel/Regularizer/l2_regularizer_grad/tuple/group_depsNoOp>^gradients/cnn1/kernel/Regularizer/l2_regularizer_grad/Reshape@^gradients/cnn1/kernel/Regularizer/l2_regularizer_grad/Reshape_1
Õ
Ngradients/cnn1/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependencyIdentity=gradients/cnn1/kernel/Regularizer/l2_regularizer_grad/ReshapeG^gradients/cnn1/kernel/Regularizer/l2_regularizer_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients/cnn1/kernel/Regularizer/l2_regularizer_grad/Reshape*
_output_shapes
: 
Ū
Pgradients/cnn1/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependency_1Identity?gradients/cnn1/kernel/Regularizer/l2_regularizer_grad/Reshape_1G^gradients/cnn1/kernel/Regularizer/l2_regularizer_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients/cnn1/kernel/Regularizer/l2_regularizer_grad/Reshape_1*
_output_shapes
: 
~
;gradients/cnn2/kernel/Regularizer/l2_regularizer_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 

=gradients/cnn2/kernel/Regularizer/l2_regularizer_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 

Kgradients/cnn2/kernel/Regularizer/l2_regularizer_grad/BroadcastGradientArgsBroadcastGradientArgs;gradients/cnn2/kernel/Regularizer/l2_regularizer_grad/Shape=gradients/cnn2/kernel/Regularizer/l2_regularizer_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
Å
9gradients/cnn2/kernel/Regularizer/l2_regularizer_grad/mulMul3gradients/full_loss_grad/tuple/control_dependency_2-cnn2/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 

9gradients/cnn2/kernel/Regularizer/l2_regularizer_grad/SumSum9gradients/cnn2/kernel/Regularizer/l2_regularizer_grad/mulKgradients/cnn2/kernel/Regularizer/l2_regularizer_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
ļ
=gradients/cnn2/kernel/Regularizer/l2_regularizer_grad/ReshapeReshape9gradients/cnn2/kernel/Regularizer/l2_regularizer_grad/Sum;gradients/cnn2/kernel/Regularizer/l2_regularizer_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
Ę
;gradients/cnn2/kernel/Regularizer/l2_regularizer_grad/mul_1Mul,cnn2/kernel/Regularizer/l2_regularizer/scale3gradients/full_loss_grad/tuple/control_dependency_2*
T0*
_output_shapes
: 

;gradients/cnn2/kernel/Regularizer/l2_regularizer_grad/Sum_1Sum;gradients/cnn2/kernel/Regularizer/l2_regularizer_grad/mul_1Mgradients/cnn2/kernel/Regularizer/l2_regularizer_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
õ
?gradients/cnn2/kernel/Regularizer/l2_regularizer_grad/Reshape_1Reshape;gradients/cnn2/kernel/Regularizer/l2_regularizer_grad/Sum_1=gradients/cnn2/kernel/Regularizer/l2_regularizer_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Š
Fgradients/cnn2/kernel/Regularizer/l2_regularizer_grad/tuple/group_depsNoOp>^gradients/cnn2/kernel/Regularizer/l2_regularizer_grad/Reshape@^gradients/cnn2/kernel/Regularizer/l2_regularizer_grad/Reshape_1
Õ
Ngradients/cnn2/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependencyIdentity=gradients/cnn2/kernel/Regularizer/l2_regularizer_grad/ReshapeG^gradients/cnn2/kernel/Regularizer/l2_regularizer_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients/cnn2/kernel/Regularizer/l2_regularizer_grad/Reshape*
_output_shapes
: 
Ū
Pgradients/cnn2/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependency_1Identity?gradients/cnn2/kernel/Regularizer/l2_regularizer_grad/Reshape_1G^gradients/cnn2/kernel/Regularizer/l2_regularizer_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients/cnn2/kernel/Regularizer/l2_regularizer_grad/Reshape_1*
_output_shapes
: 
~
;gradients/cnn3/kernel/Regularizer/l2_regularizer_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 

=gradients/cnn3/kernel/Regularizer/l2_regularizer_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 

Kgradients/cnn3/kernel/Regularizer/l2_regularizer_grad/BroadcastGradientArgsBroadcastGradientArgs;gradients/cnn3/kernel/Regularizer/l2_regularizer_grad/Shape=gradients/cnn3/kernel/Regularizer/l2_regularizer_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
Å
9gradients/cnn3/kernel/Regularizer/l2_regularizer_grad/mulMul3gradients/full_loss_grad/tuple/control_dependency_3-cnn3/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 

9gradients/cnn3/kernel/Regularizer/l2_regularizer_grad/SumSum9gradients/cnn3/kernel/Regularizer/l2_regularizer_grad/mulKgradients/cnn3/kernel/Regularizer/l2_regularizer_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
ļ
=gradients/cnn3/kernel/Regularizer/l2_regularizer_grad/ReshapeReshape9gradients/cnn3/kernel/Regularizer/l2_regularizer_grad/Sum;gradients/cnn3/kernel/Regularizer/l2_regularizer_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
Ę
;gradients/cnn3/kernel/Regularizer/l2_regularizer_grad/mul_1Mul,cnn3/kernel/Regularizer/l2_regularizer/scale3gradients/full_loss_grad/tuple/control_dependency_3*
T0*
_output_shapes
: 

;gradients/cnn3/kernel/Regularizer/l2_regularizer_grad/Sum_1Sum;gradients/cnn3/kernel/Regularizer/l2_regularizer_grad/mul_1Mgradients/cnn3/kernel/Regularizer/l2_regularizer_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
õ
?gradients/cnn3/kernel/Regularizer/l2_regularizer_grad/Reshape_1Reshape;gradients/cnn3/kernel/Regularizer/l2_regularizer_grad/Sum_1=gradients/cnn3/kernel/Regularizer/l2_regularizer_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Š
Fgradients/cnn3/kernel/Regularizer/l2_regularizer_grad/tuple/group_depsNoOp>^gradients/cnn3/kernel/Regularizer/l2_regularizer_grad/Reshape@^gradients/cnn3/kernel/Regularizer/l2_regularizer_grad/Reshape_1
Õ
Ngradients/cnn3/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependencyIdentity=gradients/cnn3/kernel/Regularizer/l2_regularizer_grad/ReshapeG^gradients/cnn3/kernel/Regularizer/l2_regularizer_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients/cnn3/kernel/Regularizer/l2_regularizer_grad/Reshape*
_output_shapes
: 
Ū
Pgradients/cnn3/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependency_1Identity?gradients/cnn3/kernel/Regularizer/l2_regularizer_grad/Reshape_1G^gradients/cnn3/kernel/Regularizer/l2_regularizer_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients/cnn3/kernel/Regularizer/l2_regularizer_grad/Reshape_1*
_output_shapes
: 
Ü
@gradients/cnn1/kernel/Regularizer/l2_regularizer/L2Loss_grad/mulMulcnn1/kernel/readPgradients/cnn1/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependency_1*
T0*&
_output_shapes
:
Ü
@gradients/cnn2/kernel/Regularizer/l2_regularizer/L2Loss_grad/mulMulcnn2/kernel/readPgradients/cnn2/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependency_1*
T0*&
_output_shapes
:0
Ü
@gradients/cnn3/kernel/Regularizer/l2_regularizer/L2Loss_grad/mulMulcnn3/kernel/readPgradients/cnn3/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependency_1*
T0*&
_output_shapes
:
q
gradients/a4_grad/Reciprocal
Reciprocala3^gradients/reward_grad/truediv*
T0*
_output_shapes
:
|
gradients/a4_grad/mulMulgradients/reward_grad/truedivgradients/a4_grad/Reciprocal*
T0*
_output_shapes
:
m
gradients/a3_grad/ShapeShapeshrink_vector*
T0*
out_type0*#
_output_shapes
:’’’’’’’’’
\
gradients/a3_grad/Shape_1Shapepn2*
T0*
out_type0*
_output_shapes
:
±
'gradients/a3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/a3_grad/Shapegradients/a3_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
[
gradients/a3_grad/mulMulgradients/a4_grad/mulpn2*
T0*
_output_shapes
:

gradients/a3_grad/SumSumgradients/a3_grad/mul'gradients/a3_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients/a3_grad/ReshapeReshapegradients/a3_grad/Sumgradients/a3_grad/Shape*
T0*
Tshape0*
_output_shapes
:
g
gradients/a3_grad/mul_1Mulshrink_vectorgradients/a4_grad/mul*
T0*
_output_shapes
:
¢
gradients/a3_grad/Sum_1Sumgradients/a3_grad/mul_1)gradients/a3_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients/a3_grad/Reshape_1Reshapegradients/a3_grad/Sum_1gradients/a3_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:’’’’’’’’’
d
"gradients/a3_grad/tuple/group_depsNoOp^gradients/a3_grad/Reshape^gradients/a3_grad/Reshape_1
Ē
*gradients/a3_grad/tuple/control_dependencyIdentitygradients/a3_grad/Reshape#^gradients/a3_grad/tuple/group_deps*
T0*,
_class"
 loc:@gradients/a3_grad/Reshape*
_output_shapes
:
Ü
,gradients/a3_grad/tuple/control_dependency_1Identitygradients/a3_grad/Reshape_1#^gradients/a3_grad/tuple/group_deps*
T0*.
_class$
" loc:@gradients/a3_grad/Reshape_1*'
_output_shapes
:’’’’’’’’’
e
"gradients/shrink_vector_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
p
$gradients/shrink_vector_grad/Shape_1ShapeMul*
T0*
out_type0*#
_output_shapes
:’’’’’’’’’
Ņ
2gradients/shrink_vector_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/shrink_vector_grad/Shape$gradients/shrink_vector_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
Ē
 gradients/shrink_vector_grad/SumSum*gradients/a3_grad/tuple/control_dependency2gradients/shrink_vector_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
¤
$gradients/shrink_vector_grad/ReshapeReshape gradients/shrink_vector_grad/Sum"gradients/shrink_vector_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
Ė
"gradients/shrink_vector_grad/Sum_1Sum*gradients/a3_grad/tuple/control_dependency4gradients/shrink_vector_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
n
 gradients/shrink_vector_grad/NegNeg"gradients/shrink_vector_grad/Sum_1*
T0*
_output_shapes
:
Ŗ
&gradients/shrink_vector_grad/Reshape_1Reshape gradients/shrink_vector_grad/Neg$gradients/shrink_vector_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:

-gradients/shrink_vector_grad/tuple/group_depsNoOp%^gradients/shrink_vector_grad/Reshape'^gradients/shrink_vector_grad/Reshape_1
ń
5gradients/shrink_vector_grad/tuple/control_dependencyIdentity$gradients/shrink_vector_grad/Reshape.^gradients/shrink_vector_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/shrink_vector_grad/Reshape*
_output_shapes
: 
ł
7gradients/shrink_vector_grad/tuple/control_dependency_1Identity&gradients/shrink_vector_grad/Reshape_1.^gradients/shrink_vector_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/shrink_vector_grad/Reshape_1*
_output_shapes
:
[
gradients/Mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
h
gradients/Mul_grad/Shape_1Shapepart2*
T0*
out_type0*#
_output_shapes
:’’’’’’’’’
“
(gradients/Mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Mul_grad/Shapegradients/Mul_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’

gradients/Mul_grad/mulMul7gradients/shrink_vector_grad/tuple/control_dependency_1part2*
T0*
_output_shapes
:

gradients/Mul_grad/SumSumgradients/Mul_grad/mul(gradients/Mul_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients/Mul_grad/ReshapeReshapegradients/Mul_grad/Sumgradients/Mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 

gradients/Mul_grad/mul_1Mulfee7gradients/shrink_vector_grad/tuple/control_dependency_1*
T0*
_output_shapes
:
„
gradients/Mul_grad/Sum_1Sumgradients/Mul_grad/mul_1*gradients/Mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients/Mul_grad/Reshape_1Reshapegradients/Mul_grad/Sum_1gradients/Mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
g
#gradients/Mul_grad/tuple/group_depsNoOp^gradients/Mul_grad/Reshape^gradients/Mul_grad/Reshape_1
É
+gradients/Mul_grad/tuple/control_dependencyIdentitygradients/Mul_grad/Reshape$^gradients/Mul_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/Mul_grad/Reshape*
_output_shapes
: 
Ń
-gradients/Mul_grad/tuple/control_dependency_1Identitygradients/Mul_grad/Reshape_1$^gradients/Mul_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Mul_grad/Reshape_1*
_output_shapes
:
f
gradients/part2_grad/ShapeShapeAbs*
T0*
out_type0*#
_output_shapes
:’’’’’’’’’

gradients/part2_grad/SizeSizegradients/part2_grad/Shape*
T0*
out_type0*-
_class#
!loc:@gradients/part2_grad/Shape*
_output_shapes
: 
£
gradients/part2_grad/addAddpart2/reduction_indicesgradients/part2_grad/Size*
T0*-
_class#
!loc:@gradients/part2_grad/Shape*
_output_shapes
: 
©
gradients/part2_grad/modFloorModgradients/part2_grad/addgradients/part2_grad/Size*
T0*-
_class#
!loc:@gradients/part2_grad/Shape*
_output_shapes
: 

gradients/part2_grad/Shape_1Const*
valueB *
dtype0*-
_class#
!loc:@gradients/part2_grad/Shape*
_output_shapes
: 

 gradients/part2_grad/range/startConst*
value	B : *
dtype0*-
_class#
!loc:@gradients/part2_grad/Shape*
_output_shapes
: 

 gradients/part2_grad/range/deltaConst*
value	B :*
dtype0*-
_class#
!loc:@gradients/part2_grad/Shape*
_output_shapes
: 
ā
gradients/part2_grad/rangeRange gradients/part2_grad/range/startgradients/part2_grad/Size gradients/part2_grad/range/delta*

Tidx0*-
_class#
!loc:@gradients/part2_grad/Shape*#
_output_shapes
:’’’’’’’’’

gradients/part2_grad/Fill/valueConst*
value	B :*
dtype0*-
_class#
!loc:@gradients/part2_grad/Shape*
_output_shapes
: 
°
gradients/part2_grad/FillFillgradients/part2_grad/Shape_1gradients/part2_grad/Fill/value*
T0*-
_class#
!loc:@gradients/part2_grad/Shape*
_output_shapes
: 

"gradients/part2_grad/DynamicStitchDynamicStitchgradients/part2_grad/rangegradients/part2_grad/modgradients/part2_grad/Shapegradients/part2_grad/Fill*
N*
T0*-
_class#
!loc:@gradients/part2_grad/Shape*#
_output_shapes
:’’’’’’’’’

gradients/part2_grad/Maximum/yConst*
value	B :*
dtype0*-
_class#
!loc:@gradients/part2_grad/Shape*
_output_shapes
: 
Č
gradients/part2_grad/MaximumMaximum"gradients/part2_grad/DynamicStitchgradients/part2_grad/Maximum/y*
T0*-
_class#
!loc:@gradients/part2_grad/Shape*#
_output_shapes
:’’’’’’’’’
Ą
gradients/part2_grad/floordivFloorDivgradients/part2_grad/Shapegradients/part2_grad/Maximum*
T0*-
_class#
!loc:@gradients/part2_grad/Shape*#
_output_shapes
:’’’’’’’’’
«
gradients/part2_grad/ReshapeReshape-gradients/Mul_grad/tuple/control_dependency_1"gradients/part2_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:

gradients/part2_grad/TileTilegradients/part2_grad/Reshapegradients/part2_grad/floordiv*
T0*

Tmultiples0*
_output_shapes
:
I
gradients/Abs_grad/SignSignpart1*
T0*
_output_shapes
:
t
gradients/Abs_grad/mulMulgradients/part2_grad/Tilegradients/Abs_grad/Sign*
T0*
_output_shapes
:
i
gradients/part1_grad/ShapeShapestrided_slice_1*
T0*
out_type0*
_output_shapes
:
t
gradients/part1_grad/Shape_1Shapestrided_slice_2*
T0*
out_type0*#
_output_shapes
:’’’’’’’’’
ŗ
*gradients/part1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/part1_grad/Shapegradients/part1_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
£
gradients/part1_grad/SumSumgradients/Abs_grad/mul*gradients/part1_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients/part1_grad/ReshapeReshapegradients/part1_grad/Sumgradients/part1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:’’’’’’’’’
§
gradients/part1_grad/Sum_1Sumgradients/Abs_grad/mul,gradients/part1_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
^
gradients/part1_grad/NegNeggradients/part1_grad/Sum_1*
T0*
_output_shapes
:

gradients/part1_grad/Reshape_1Reshapegradients/part1_grad/Neggradients/part1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
m
%gradients/part1_grad/tuple/group_depsNoOp^gradients/part1_grad/Reshape^gradients/part1_grad/Reshape_1
ā
-gradients/part1_grad/tuple/control_dependencyIdentitygradients/part1_grad/Reshape&^gradients/part1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/part1_grad/Reshape*'
_output_shapes
:’’’’’’’’’
Ł
/gradients/part1_grad/tuple/control_dependency_1Identitygradients/part1_grad/Reshape_1&^gradients/part1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/part1_grad/Reshape_1*
_output_shapes
:
u
$gradients/strided_slice_1_grad/ShapeShapecurrent_portfolio*
T0*
out_type0*
_output_shapes
:

/gradients/strided_slice_1_grad/StridedSliceGradStridedSliceGrad$gradients/strided_slice_1_grad/Shapestrided_slice_1/stackstrided_slice_1/stack_1strided_slice_1/stack_2-gradients/part1_grad/tuple/control_dependency*
T0*
Index0*

begin_mask*
end_mask*
ellipsis_mask *
new_axis_mask *
shrink_axis_mask *'
_output_shapes
:’’’’’’’’’

$gradients/strided_slice_2_grad/ShapeShapecalculated_portfolio*
T0*
out_type0*#
_output_shapes
:’’’’’’’’’
÷
/gradients/strided_slice_2_grad/StridedSliceGradStridedSliceGrad$gradients/strided_slice_2_grad/Shapestrided_slice_2/stackstrided_slice_2/stack_1strided_slice_2/stack_2/gradients/part1_grad/tuple/control_dependency_1*
T0*
Index0*

begin_mask*
end_mask*
ellipsis_mask *
new_axis_mask *
shrink_axis_mask *
_output_shapes
:
i
&gradients/current_portfolio_grad/ShapeShapepn1*
T0*
out_type0*
_output_shapes
:
k
(gradients/current_portfolio_grad/Shape_1Shapepn2*
T0*
out_type0*
_output_shapes
:
Ž
6gradients/current_portfolio_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/current_portfolio_grad/Shape(gradients/current_portfolio_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’

(gradients/current_portfolio_grad/RealDivRealDiv/gradients/strided_slice_1_grad/StridedSliceGradpn2*
T0*'
_output_shapes
:’’’’’’’’’
Ķ
$gradients/current_portfolio_grad/SumSum(gradients/current_portfolio_grad/RealDiv6gradients/current_portfolio_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
Į
(gradients/current_portfolio_grad/ReshapeReshape$gradients/current_portfolio_grad/Sum&gradients/current_portfolio_grad/Shape*
T0*
Tshape0*'
_output_shapes
:’’’’’’’’’
b
$gradients/current_portfolio_grad/NegNegpn1*
T0*'
_output_shapes
:’’’’’’’’’

*gradients/current_portfolio_grad/RealDiv_1RealDiv$gradients/current_portfolio_grad/Negpn2*
T0*'
_output_shapes
:’’’’’’’’’

*gradients/current_portfolio_grad/RealDiv_2RealDiv*gradients/current_portfolio_grad/RealDiv_1pn2*
T0*'
_output_shapes
:’’’’’’’’’
ŗ
$gradients/current_portfolio_grad/mulMul/gradients/strided_slice_1_grad/StridedSliceGrad*gradients/current_portfolio_grad/RealDiv_2*
T0*'
_output_shapes
:’’’’’’’’’
Ķ
&gradients/current_portfolio_grad/Sum_1Sum$gradients/current_portfolio_grad/mul8gradients/current_portfolio_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
Ē
*gradients/current_portfolio_grad/Reshape_1Reshape&gradients/current_portfolio_grad/Sum_1(gradients/current_portfolio_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:’’’’’’’’’

1gradients/current_portfolio_grad/tuple/group_depsNoOp)^gradients/current_portfolio_grad/Reshape+^gradients/current_portfolio_grad/Reshape_1

9gradients/current_portfolio_grad/tuple/control_dependencyIdentity(gradients/current_portfolio_grad/Reshape2^gradients/current_portfolio_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/current_portfolio_grad/Reshape*'
_output_shapes
:’’’’’’’’’

;gradients/current_portfolio_grad/tuple/control_dependency_1Identity*gradients/current_portfolio_grad/Reshape_12^gradients/current_portfolio_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/current_portfolio_grad/Reshape_1*'
_output_shapes
:’’’’’’’’’
ģ
gradients/AddNAddN,gradients/a3_grad/tuple/control_dependency_1;gradients/current_portfolio_grad/tuple/control_dependency_1*
N*
T0*.
_class$
" loc:@gradients/a3_grad/Reshape_1*'
_output_shapes
:’’’’’’’’’
[
gradients/pn2_grad/ShapeShapepn1*
T0*
out_type0*
_output_shapes
:

gradients/pn2_grad/SizeConst*
value	B :*
dtype0*+
_class!
loc:@gradients/pn2_grad/Shape*
_output_shapes
: 

gradients/pn2_grad/addAddpn2/reduction_indicesgradients/pn2_grad/Size*
T0*+
_class!
loc:@gradients/pn2_grad/Shape*
_output_shapes
: 
”
gradients/pn2_grad/modFloorModgradients/pn2_grad/addgradients/pn2_grad/Size*
T0*+
_class!
loc:@gradients/pn2_grad/Shape*
_output_shapes
: 

gradients/pn2_grad/Shape_1Const*
valueB *
dtype0*+
_class!
loc:@gradients/pn2_grad/Shape*
_output_shapes
: 

gradients/pn2_grad/range/startConst*
value	B : *
dtype0*+
_class!
loc:@gradients/pn2_grad/Shape*
_output_shapes
: 

gradients/pn2_grad/range/deltaConst*
value	B :*
dtype0*+
_class!
loc:@gradients/pn2_grad/Shape*
_output_shapes
: 
Ļ
gradients/pn2_grad/rangeRangegradients/pn2_grad/range/startgradients/pn2_grad/Sizegradients/pn2_grad/range/delta*

Tidx0*+
_class!
loc:@gradients/pn2_grad/Shape*
_output_shapes
:

gradients/pn2_grad/Fill/valueConst*
value	B :*
dtype0*+
_class!
loc:@gradients/pn2_grad/Shape*
_output_shapes
: 
Ø
gradients/pn2_grad/FillFillgradients/pn2_grad/Shape_1gradients/pn2_grad/Fill/value*
T0*+
_class!
loc:@gradients/pn2_grad/Shape*
_output_shapes
: 
ś
 gradients/pn2_grad/DynamicStitchDynamicStitchgradients/pn2_grad/rangegradients/pn2_grad/modgradients/pn2_grad/Shapegradients/pn2_grad/Fill*
N*
T0*+
_class!
loc:@gradients/pn2_grad/Shape*#
_output_shapes
:’’’’’’’’’

gradients/pn2_grad/Maximum/yConst*
value	B :*
dtype0*+
_class!
loc:@gradients/pn2_grad/Shape*
_output_shapes
: 
Ą
gradients/pn2_grad/MaximumMaximum gradients/pn2_grad/DynamicStitchgradients/pn2_grad/Maximum/y*
T0*+
_class!
loc:@gradients/pn2_grad/Shape*#
_output_shapes
:’’’’’’’’’
Æ
gradients/pn2_grad/floordivFloorDivgradients/pn2_grad/Shapegradients/pn2_grad/Maximum*
T0*+
_class!
loc:@gradients/pn2_grad/Shape*
_output_shapes
:

gradients/pn2_grad/ReshapeReshapegradients/AddN gradients/pn2_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:

gradients/pn2_grad/TileTilegradients/pn2_grad/Reshapegradients/pn2_grad/floordiv*
T0*

Tmultiples0*'
_output_shapes
:’’’’’’’’’
ä
gradients/AddN_1AddN9gradients/current_portfolio_grad/tuple/control_dependencygradients/pn2_grad/Tile*
N*
T0*;
_class1
/-loc:@gradients/current_portfolio_grad/Reshape*'
_output_shapes
:’’’’’’’’’
Y
gradients/pn1_grad/ShapeShapey*
T0*
out_type0*
_output_shapes
:
q
gradients/pn1_grad/Shape_1Shapeprevious_step_portfolio*
T0*
out_type0*
_output_shapes
:
“
(gradients/pn1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pn1_grad/Shapegradients/pn1_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
z
gradients/pn1_grad/mulMulgradients/AddN_1previous_step_portfolio*
T0*'
_output_shapes
:’’’’’’’’’

gradients/pn1_grad/SumSumgradients/pn1_grad/mul(gradients/pn1_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients/pn1_grad/ReshapeReshapegradients/pn1_grad/Sumgradients/pn1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:’’’’’’’’’
f
gradients/pn1_grad/mul_1Mulygradients/AddN_1*
T0*'
_output_shapes
:’’’’’’’’’
„
gradients/pn1_grad/Sum_1Sumgradients/pn1_grad/mul_1*gradients/pn1_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients/pn1_grad/Reshape_1Reshapegradients/pn1_grad/Sum_1gradients/pn1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:’’’’’’’’’
g
#gradients/pn1_grad/tuple/group_depsNoOp^gradients/pn1_grad/Reshape^gradients/pn1_grad/Reshape_1
Ś
+gradients/pn1_grad/tuple/control_dependencyIdentitygradients/pn1_grad/Reshape$^gradients/pn1_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/pn1_grad/Reshape*'
_output_shapes
:’’’’’’’’’
ą
-gradients/pn1_grad/tuple/control_dependency_1Identitygradients/pn1_grad/Reshape_1$^gradients/pn1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/pn1_grad/Reshape_1*'
_output_shapes
:’’’’’’’’’
m
+gradients/previous_step_portfolio_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
¢
*gradients/previous_step_portfolio_grad/modFloorModprevious_step_portfolio/axis+gradients/previous_step_portfolio_grad/Rank*
T0*
_output_shapes
: 
}
,gradients/previous_step_portfolio_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:

.gradients/previous_step_portfolio_grad/Shape_1Shapestrided_slice*
T0*
out_type0*#
_output_shapes
:’’’’’’’’’
”
-gradients/previous_step_portfolio_grad/ShapeNShapeNbuffer_valuestrided_slice*
N*
T0*
out_type0*)
_output_shapes
::’’’’’’’’’

3gradients/previous_step_portfolio_grad/ConcatOffsetConcatOffset*gradients/previous_step_portfolio_grad/mod-gradients/previous_step_portfolio_grad/ShapeN/gradients/previous_step_portfolio_grad/ShapeN:1*
N*)
_output_shapes
::’’’’’’’’’

,gradients/previous_step_portfolio_grad/SliceSlice-gradients/pn1_grad/tuple/control_dependency_13gradients/previous_step_portfolio_grad/ConcatOffset-gradients/previous_step_portfolio_grad/ShapeN*
T0*
Index0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’

.gradients/previous_step_portfolio_grad/Slice_1Slice-gradients/pn1_grad/tuple/control_dependency_15gradients/previous_step_portfolio_grad/ConcatOffset:1/gradients/previous_step_portfolio_grad/ShapeN:1*
T0*
Index0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’

7gradients/previous_step_portfolio_grad/tuple/group_depsNoOp-^gradients/previous_step_portfolio_grad/Slice/^gradients/previous_step_portfolio_grad/Slice_1

?gradients/previous_step_portfolio_grad/tuple/control_dependencyIdentity,gradients/previous_step_portfolio_grad/Slice8^gradients/previous_step_portfolio_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/previous_step_portfolio_grad/Slice*
_output_shapes

:
µ
Agradients/previous_step_portfolio_grad/tuple/control_dependency_1Identity.gradients/previous_step_portfolio_grad/Slice_18^gradients/previous_step_portfolio_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/previous_step_portfolio_grad/Slice_1*0
_output_shapes
:’’’’’’’’’’’’’’’’’’

"gradients/strided_slice_grad/ShapeShapecalculated_portfolio*
T0*
out_type0*#
_output_shapes
:’’’’’’’’’
’
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad"gradients/strided_slice_grad/Shapestrided_slice/stackstrided_slice/stack_1strided_slice/stack_2Agradients/previous_step_portfolio_grad/tuple/control_dependency_1*
T0*
Index0*

begin_mask *
end_mask *
ellipsis_mask *
new_axis_mask *
shrink_axis_mask *
_output_shapes
:
č
gradients/AddN_2AddN/gradients/strided_slice_2_grad/StridedSliceGrad-gradients/strided_slice_grad/StridedSliceGrad*
N*
T0*B
_class8
64loc:@gradients/strided_slice_2_grad/StridedSliceGrad*
_output_shapes
:
p
)gradients/calculated_portfolio_grad/ShapeShapeSoftmax*
T0*
out_type0*
_output_shapes
:
¼
+gradients/calculated_portfolio_grad/ReshapeReshapegradients/AddN_2)gradients/calculated_portfolio_grad/Shape*
T0*
Tshape0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’

gradients/Softmax_grad/mulMul+gradients/calculated_portfolio_grad/ReshapeSoftmax*
T0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’
v
,gradients/Softmax_grad/Sum/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
¶
gradients/Softmax_grad/SumSumgradients/Softmax_grad/mul,gradients/Softmax_grad/Sum/reduction_indices*
	keep_dims( *
T0*

Tidx0*#
_output_shapes
:’’’’’’’’’
u
$gradients/Softmax_grad/Reshape/shapeConst*
valueB"’’’’   *
dtype0*
_output_shapes
:
«
gradients/Softmax_grad/ReshapeReshapegradients/Softmax_grad/Sum$gradients/Softmax_grad/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:’’’’’’’’’
©
gradients/Softmax_grad/subSub+gradients/calculated_portfolio_grad/Reshapegradients/Softmax_grad/Reshape*
T0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’

gradients/Softmax_grad/mul_1Mulgradients/Softmax_grad/subSoftmax*
T0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’
l
gradients/Reshape_grad/ShapeShapeSqueeze*
T0*
out_type0*#
_output_shapes
:’’’’’’’’’

gradients/Reshape_grad/ReshapeReshapegradients/Softmax_grad/mul_1gradients/Reshape_grad/Shape*
T0*
Tshape0*
_output_shapes
:
h
gradients/Squeeze_grad/ShapeShapecnn3/BiasAdd*
T0*
out_type0*
_output_shapes
:
Æ
gradients/Squeeze_grad/ReshapeReshapegradients/Reshape_grad/Reshapegradients/Squeeze_grad/Shape*
T0*
Tshape0*/
_output_shapes
:’’’’’’’’’

'gradients/cnn3/BiasAdd_grad/BiasAddGradBiasAddGradgradients/Squeeze_grad/Reshape*
T0*
data_formatNHWC*
_output_shapes
:

,gradients/cnn3/BiasAdd_grad/tuple/group_depsNoOp^gradients/Squeeze_grad/Reshape(^gradients/cnn3/BiasAdd_grad/BiasAddGrad
ü
4gradients/cnn3/BiasAdd_grad/tuple/control_dependencyIdentitygradients/Squeeze_grad/Reshape-^gradients/cnn3/BiasAdd_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Squeeze_grad/Reshape*/
_output_shapes
:’’’’’’’’’
ū
6gradients/cnn3/BiasAdd_grad/tuple/control_dependency_1Identity'gradients/cnn3/BiasAdd_grad/BiasAddGrad-^gradients/cnn3/BiasAdd_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/cnn3/BiasAdd_grad/BiasAddGrad*
_output_shapes
:

!gradients/cnn3/Conv2D_grad/ShapeNShapeNcnn2/Elucnn3/kernel/read*
N*
T0*
out_type0* 
_output_shapes
::
Ż
.gradients/cnn3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput!gradients/cnn3/Conv2D_grad/ShapeNcnn3/kernel/read4gradients/cnn3/BiasAdd_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingVALID*
data_formatNHWC*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ł
/gradients/cnn3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltercnn2/Elu#gradients/cnn3/Conv2D_grad/ShapeN:14gradients/cnn3/BiasAdd_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingVALID*
data_formatNHWC*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’

+gradients/cnn3/Conv2D_grad/tuple/group_depsNoOp/^gradients/cnn3/Conv2D_grad/Conv2DBackpropInput0^gradients/cnn3/Conv2D_grad/Conv2DBackpropFilter

3gradients/cnn3/Conv2D_grad/tuple/control_dependencyIdentity.gradients/cnn3/Conv2D_grad/Conv2DBackpropInput,^gradients/cnn3/Conv2D_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/cnn3/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:’’’’’’’’’

5gradients/cnn3/Conv2D_grad/tuple/control_dependency_1Identity/gradients/cnn3/Conv2D_grad/Conv2DBackpropFilter,^gradients/cnn3/Conv2D_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/cnn3/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:
£
gradients/cnn2/Elu_grad/EluGradEluGrad3gradients/cnn3/Conv2D_grad/tuple/control_dependencycnn2/Elu*
T0*/
_output_shapes
:’’’’’’’’’
 
gradients/AddN_3AddN@gradients/cnn3/kernel/Regularizer/l2_regularizer/L2Loss_grad/mul5gradients/cnn3/Conv2D_grad/tuple/control_dependency_1*
N*
T0*S
_classI
GEloc:@gradients/cnn3/kernel/Regularizer/l2_regularizer/L2Loss_grad/mul*&
_output_shapes
:

'gradients/cnn2/BiasAdd_grad/BiasAddGradBiasAddGradgradients/cnn2/Elu_grad/EluGrad*
T0*
data_formatNHWC*
_output_shapes
:

,gradients/cnn2/BiasAdd_grad/tuple/group_depsNoOp ^gradients/cnn2/Elu_grad/EluGrad(^gradients/cnn2/BiasAdd_grad/BiasAddGrad
ž
4gradients/cnn2/BiasAdd_grad/tuple/control_dependencyIdentitygradients/cnn2/Elu_grad/EluGrad-^gradients/cnn2/BiasAdd_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/cnn2/Elu_grad/EluGrad*/
_output_shapes
:’’’’’’’’’
ū
6gradients/cnn2/BiasAdd_grad/tuple/control_dependency_1Identity'gradients/cnn2/BiasAdd_grad/BiasAddGrad-^gradients/cnn2/BiasAdd_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/cnn2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:

!gradients/cnn2/Conv2D_grad/ShapeNShapeNcnn1/Elucnn2/kernel/read*
N*
T0*
out_type0* 
_output_shapes
::
Ż
.gradients/cnn2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput!gradients/cnn2/Conv2D_grad/ShapeNcnn2/kernel/read4gradients/cnn2/BiasAdd_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingVALID*
data_formatNHWC*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ł
/gradients/cnn2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltercnn1/Elu#gradients/cnn2/Conv2D_grad/ShapeN:14gradients/cnn2/BiasAdd_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingVALID*
data_formatNHWC*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’

+gradients/cnn2/Conv2D_grad/tuple/group_depsNoOp/^gradients/cnn2/Conv2D_grad/Conv2DBackpropInput0^gradients/cnn2/Conv2D_grad/Conv2DBackpropFilter

3gradients/cnn2/Conv2D_grad/tuple/control_dependencyIdentity.gradients/cnn2/Conv2D_grad/Conv2DBackpropInput,^gradients/cnn2/Conv2D_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/cnn2/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:’’’’’’’’’0

5gradients/cnn2/Conv2D_grad/tuple/control_dependency_1Identity/gradients/cnn2/Conv2D_grad/Conv2DBackpropFilter,^gradients/cnn2/Conv2D_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/cnn2/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:0
£
gradients/cnn1/Elu_grad/EluGradEluGrad3gradients/cnn2/Conv2D_grad/tuple/control_dependencycnn1/Elu*
T0*/
_output_shapes
:’’’’’’’’’0
 
gradients/AddN_4AddN@gradients/cnn2/kernel/Regularizer/l2_regularizer/L2Loss_grad/mul5gradients/cnn2/Conv2D_grad/tuple/control_dependency_1*
N*
T0*S
_classI
GEloc:@gradients/cnn2/kernel/Regularizer/l2_regularizer/L2Loss_grad/mul*&
_output_shapes
:0

'gradients/cnn1/BiasAdd_grad/BiasAddGradBiasAddGradgradients/cnn1/Elu_grad/EluGrad*
T0*
data_formatNHWC*
_output_shapes
:

,gradients/cnn1/BiasAdd_grad/tuple/group_depsNoOp ^gradients/cnn1/Elu_grad/EluGrad(^gradients/cnn1/BiasAdd_grad/BiasAddGrad
ž
4gradients/cnn1/BiasAdd_grad/tuple/control_dependencyIdentitygradients/cnn1/Elu_grad/EluGrad-^gradients/cnn1/BiasAdd_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/cnn1/Elu_grad/EluGrad*/
_output_shapes
:’’’’’’’’’0
ū
6gradients/cnn1/BiasAdd_grad/tuple/control_dependency_1Identity'gradients/cnn1/BiasAdd_grad/BiasAddGrad-^gradients/cnn1/BiasAdd_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/cnn1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:

!gradients/cnn1/Conv2D_grad/ShapeNShapeNXcnn1/kernel/read*
N*
T0*
out_type0* 
_output_shapes
::
Ż
.gradients/cnn1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput!gradients/cnn1/Conv2D_grad/ShapeNcnn1/kernel/read4gradients/cnn1/BiasAdd_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingVALID*
data_formatNHWC*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ņ
/gradients/cnn1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterX#gradients/cnn1/Conv2D_grad/ShapeN:14gradients/cnn1/BiasAdd_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingVALID*
data_formatNHWC*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’

+gradients/cnn1/Conv2D_grad/tuple/group_depsNoOp/^gradients/cnn1/Conv2D_grad/Conv2DBackpropInput0^gradients/cnn1/Conv2D_grad/Conv2DBackpropFilter

3gradients/cnn1/Conv2D_grad/tuple/control_dependencyIdentity.gradients/cnn1/Conv2D_grad/Conv2DBackpropInput,^gradients/cnn1/Conv2D_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/cnn1/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:’’’’’’’’’2

5gradients/cnn1/Conv2D_grad/tuple/control_dependency_1Identity/gradients/cnn1/Conv2D_grad/Conv2DBackpropFilter,^gradients/cnn1/Conv2D_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/cnn1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:
 
gradients/AddN_5AddN@gradients/cnn1/kernel/Regularizer/l2_regularizer/L2Loss_grad/mul5gradients/cnn1/Conv2D_grad/tuple/control_dependency_1*
N*
T0*S
_classI
GEloc:@gradients/cnn1/kernel/Regularizer/l2_regularizer/L2Loss_grad/mul*&
_output_shapes
:
|
beta1_power/initial_valueConst*
valueB
 *fff?*
dtype0*
_class
loc:@cnn1/bias*
_output_shapes
: 

beta1_power
VariableV2*
shape: *
dtype0*
	container *
shared_name *
_class
loc:@cnn1/bias*
_output_shapes
: 
¬
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn1/bias*
_output_shapes
: 
h
beta1_power/readIdentitybeta1_power*
T0*
_class
loc:@cnn1/bias*
_output_shapes
: 
|
beta2_power/initial_valueConst*
valueB
 *w¾?*
dtype0*
_class
loc:@cnn1/bias*
_output_shapes
: 

beta2_power
VariableV2*
shape: *
dtype0*
	container *
shared_name *
_class
loc:@cnn1/bias*
_output_shapes
: 
¬
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn1/bias*
_output_shapes
: 
h
beta2_power/readIdentitybeta2_power*
T0*
_class
loc:@cnn1/bias*
_output_shapes
: 
§
"cnn1/kernel/Adam/Initializer/zerosConst*%
valueB*    *
dtype0*
_class
loc:@cnn1/kernel*&
_output_shapes
:
“
cnn1/kernel/Adam
VariableV2*
shape:*
dtype0*
	container *
shared_name *
_class
loc:@cnn1/kernel*&
_output_shapes
:
Ń
cnn1/kernel/Adam/AssignAssigncnn1/kernel/Adam"cnn1/kernel/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn1/kernel*&
_output_shapes
:

cnn1/kernel/Adam/readIdentitycnn1/kernel/Adam*
T0*
_class
loc:@cnn1/kernel*&
_output_shapes
:
©
$cnn1/kernel/Adam_1/Initializer/zerosConst*%
valueB*    *
dtype0*
_class
loc:@cnn1/kernel*&
_output_shapes
:
¶
cnn1/kernel/Adam_1
VariableV2*
shape:*
dtype0*
	container *
shared_name *
_class
loc:@cnn1/kernel*&
_output_shapes
:
×
cnn1/kernel/Adam_1/AssignAssigncnn1/kernel/Adam_1$cnn1/kernel/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn1/kernel*&
_output_shapes
:

cnn1/kernel/Adam_1/readIdentitycnn1/kernel/Adam_1*
T0*
_class
loc:@cnn1/kernel*&
_output_shapes
:

 cnn1/bias/Adam/Initializer/zerosConst*
valueB*    *
dtype0*
_class
loc:@cnn1/bias*
_output_shapes
:

cnn1/bias/Adam
VariableV2*
shape:*
dtype0*
	container *
shared_name *
_class
loc:@cnn1/bias*
_output_shapes
:
½
cnn1/bias/Adam/AssignAssigncnn1/bias/Adam cnn1/bias/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn1/bias*
_output_shapes
:
r
cnn1/bias/Adam/readIdentitycnn1/bias/Adam*
T0*
_class
loc:@cnn1/bias*
_output_shapes
:

"cnn1/bias/Adam_1/Initializer/zerosConst*
valueB*    *
dtype0*
_class
loc:@cnn1/bias*
_output_shapes
:

cnn1/bias/Adam_1
VariableV2*
shape:*
dtype0*
	container *
shared_name *
_class
loc:@cnn1/bias*
_output_shapes
:
Ć
cnn1/bias/Adam_1/AssignAssigncnn1/bias/Adam_1"cnn1/bias/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn1/bias*
_output_shapes
:
v
cnn1/bias/Adam_1/readIdentitycnn1/bias/Adam_1*
T0*
_class
loc:@cnn1/bias*
_output_shapes
:
§
"cnn2/kernel/Adam/Initializer/zerosConst*%
valueB0*    *
dtype0*
_class
loc:@cnn2/kernel*&
_output_shapes
:0
“
cnn2/kernel/Adam
VariableV2*
shape:0*
dtype0*
	container *
shared_name *
_class
loc:@cnn2/kernel*&
_output_shapes
:0
Ń
cnn2/kernel/Adam/AssignAssigncnn2/kernel/Adam"cnn2/kernel/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn2/kernel*&
_output_shapes
:0

cnn2/kernel/Adam/readIdentitycnn2/kernel/Adam*
T0*
_class
loc:@cnn2/kernel*&
_output_shapes
:0
©
$cnn2/kernel/Adam_1/Initializer/zerosConst*%
valueB0*    *
dtype0*
_class
loc:@cnn2/kernel*&
_output_shapes
:0
¶
cnn2/kernel/Adam_1
VariableV2*
shape:0*
dtype0*
	container *
shared_name *
_class
loc:@cnn2/kernel*&
_output_shapes
:0
×
cnn2/kernel/Adam_1/AssignAssigncnn2/kernel/Adam_1$cnn2/kernel/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn2/kernel*&
_output_shapes
:0

cnn2/kernel/Adam_1/readIdentitycnn2/kernel/Adam_1*
T0*
_class
loc:@cnn2/kernel*&
_output_shapes
:0

 cnn2/bias/Adam/Initializer/zerosConst*
valueB*    *
dtype0*
_class
loc:@cnn2/bias*
_output_shapes
:

cnn2/bias/Adam
VariableV2*
shape:*
dtype0*
	container *
shared_name *
_class
loc:@cnn2/bias*
_output_shapes
:
½
cnn2/bias/Adam/AssignAssigncnn2/bias/Adam cnn2/bias/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn2/bias*
_output_shapes
:
r
cnn2/bias/Adam/readIdentitycnn2/bias/Adam*
T0*
_class
loc:@cnn2/bias*
_output_shapes
:

"cnn2/bias/Adam_1/Initializer/zerosConst*
valueB*    *
dtype0*
_class
loc:@cnn2/bias*
_output_shapes
:

cnn2/bias/Adam_1
VariableV2*
shape:*
dtype0*
	container *
shared_name *
_class
loc:@cnn2/bias*
_output_shapes
:
Ć
cnn2/bias/Adam_1/AssignAssigncnn2/bias/Adam_1"cnn2/bias/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn2/bias*
_output_shapes
:
v
cnn2/bias/Adam_1/readIdentitycnn2/bias/Adam_1*
T0*
_class
loc:@cnn2/bias*
_output_shapes
:
§
"cnn3/kernel/Adam/Initializer/zerosConst*%
valueB*    *
dtype0*
_class
loc:@cnn3/kernel*&
_output_shapes
:
“
cnn3/kernel/Adam
VariableV2*
shape:*
dtype0*
	container *
shared_name *
_class
loc:@cnn3/kernel*&
_output_shapes
:
Ń
cnn3/kernel/Adam/AssignAssigncnn3/kernel/Adam"cnn3/kernel/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn3/kernel*&
_output_shapes
:

cnn3/kernel/Adam/readIdentitycnn3/kernel/Adam*
T0*
_class
loc:@cnn3/kernel*&
_output_shapes
:
©
$cnn3/kernel/Adam_1/Initializer/zerosConst*%
valueB*    *
dtype0*
_class
loc:@cnn3/kernel*&
_output_shapes
:
¶
cnn3/kernel/Adam_1
VariableV2*
shape:*
dtype0*
	container *
shared_name *
_class
loc:@cnn3/kernel*&
_output_shapes
:
×
cnn3/kernel/Adam_1/AssignAssigncnn3/kernel/Adam_1$cnn3/kernel/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn3/kernel*&
_output_shapes
:

cnn3/kernel/Adam_1/readIdentitycnn3/kernel/Adam_1*
T0*
_class
loc:@cnn3/kernel*&
_output_shapes
:

 cnn3/bias/Adam/Initializer/zerosConst*
valueB*    *
dtype0*
_class
loc:@cnn3/bias*
_output_shapes
:

cnn3/bias/Adam
VariableV2*
shape:*
dtype0*
	container *
shared_name *
_class
loc:@cnn3/bias*
_output_shapes
:
½
cnn3/bias/Adam/AssignAssigncnn3/bias/Adam cnn3/bias/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn3/bias*
_output_shapes
:
r
cnn3/bias/Adam/readIdentitycnn3/bias/Adam*
T0*
_class
loc:@cnn3/bias*
_output_shapes
:

"cnn3/bias/Adam_1/Initializer/zerosConst*
valueB*    *
dtype0*
_class
loc:@cnn3/bias*
_output_shapes
:

cnn3/bias/Adam_1
VariableV2*
shape:*
dtype0*
	container *
shared_name *
_class
loc:@cnn3/bias*
_output_shapes
:
Ć
cnn3/bias/Adam_1/AssignAssigncnn3/bias/Adam_1"cnn3/bias/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn3/bias*
_output_shapes
:
v
cnn3/bias/Adam_1/readIdentitycnn3/bias/Adam_1*
T0*
_class
loc:@cnn3/bias*
_output_shapes
:
O

Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
O

Adam/beta2Const*
valueB
 *w¾?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
valueB
 *wĢ+2*
dtype0*
_output_shapes
: 
Ē
!Adam/update_cnn1/kernel/ApplyAdam	ApplyAdamcnn1/kernelcnn1/kernel/Adamcnn1/kernel/Adam_1beta1_power/readbeta2_power/readExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_5*
T0*
use_locking( *
use_nesterov( *
_class
loc:@cnn1/kernel*&
_output_shapes
:
×
Adam/update_cnn1/bias/ApplyAdam	ApplyAdam	cnn1/biascnn1/bias/Adamcnn1/bias/Adam_1beta1_power/readbeta2_power/readExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilon6gradients/cnn1/BiasAdd_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *
_class
loc:@cnn1/bias*
_output_shapes
:
Ē
!Adam/update_cnn2/kernel/ApplyAdam	ApplyAdamcnn2/kernelcnn2/kernel/Adamcnn2/kernel/Adam_1beta1_power/readbeta2_power/readExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_4*
T0*
use_locking( *
use_nesterov( *
_class
loc:@cnn2/kernel*&
_output_shapes
:0
×
Adam/update_cnn2/bias/ApplyAdam	ApplyAdam	cnn2/biascnn2/bias/Adamcnn2/bias/Adam_1beta1_power/readbeta2_power/readExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilon6gradients/cnn2/BiasAdd_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *
_class
loc:@cnn2/bias*
_output_shapes
:
Ē
!Adam/update_cnn3/kernel/ApplyAdam	ApplyAdamcnn3/kernelcnn3/kernel/Adamcnn3/kernel/Adam_1beta1_power/readbeta2_power/readExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_3*
T0*
use_locking( *
use_nesterov( *
_class
loc:@cnn3/kernel*&
_output_shapes
:
×
Adam/update_cnn3/bias/ApplyAdam	ApplyAdam	cnn3/biascnn3/bias/Adamcnn3/bias/Adam_1beta1_power/readbeta2_power/readExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilon6gradients/cnn3/BiasAdd_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *
_class
loc:@cnn3/bias*
_output_shapes
:
¾
Adam/mulMulbeta1_power/read
Adam/beta1"^Adam/update_cnn1/kernel/ApplyAdam ^Adam/update_cnn1/bias/ApplyAdam"^Adam/update_cnn2/kernel/ApplyAdam ^Adam/update_cnn2/bias/ApplyAdam"^Adam/update_cnn3/kernel/ApplyAdam ^Adam/update_cnn3/bias/ApplyAdam*
T0*
_class
loc:@cnn1/bias*
_output_shapes
: 

Adam/AssignAssignbeta1_powerAdam/mul*
T0*
validate_shape(*
use_locking( *
_class
loc:@cnn1/bias*
_output_shapes
: 
Ą

Adam/mul_1Mulbeta2_power/read
Adam/beta2"^Adam/update_cnn1/kernel/ApplyAdam ^Adam/update_cnn1/bias/ApplyAdam"^Adam/update_cnn2/kernel/ApplyAdam ^Adam/update_cnn2/bias/ApplyAdam"^Adam/update_cnn3/kernel/ApplyAdam ^Adam/update_cnn3/bias/ApplyAdam*
T0*
_class
loc:@cnn1/bias*
_output_shapes
: 

Adam/Assign_1Assignbeta2_power
Adam/mul_1*
T0*
validate_shape(*
use_locking( *
_class
loc:@cnn1/bias*
_output_shapes
: 

Adam/updateNoOp"^Adam/update_cnn1/kernel/ApplyAdam ^Adam/update_cnn1/bias/ApplyAdam"^Adam/update_cnn2/kernel/ApplyAdam ^Adam/update_cnn2/bias/ApplyAdam"^Adam/update_cnn3/kernel/ApplyAdam ^Adam/update_cnn3/bias/ApplyAdam^Adam/Assign^Adam/Assign_1
w

Adam/valueConst^Adam/update*
value	B :*
dtype0*
_class
loc:@Variable*
_output_shapes
: 
x
Adam	AssignAddVariable
Adam/value*
T0*
use_locking( *
_class
loc:@Variable*
_output_shapes
: 
Å
Merge/MergeSummaryMergeSummary	portfoliocnn1/kernel_1cnn1/bias_1cnn2/kernel_1cnn2/bias_1cnn3/kernel_1cnn3/bias_1mu	base_lossfull_loss_1sum_p*
N*
_output_shapes
: 
ų
initNoOp^cnn1/kernel/Assign^cnn1/bias/Assign^cnn2/kernel/Assign^cnn2/bias/Assign^cnn3/kernel/Assign^cnn3/bias/Assign^Variable/Assign^beta1_power/Assign^beta2_power/Assign^cnn1/kernel/Adam/Assign^cnn1/kernel/Adam_1/Assign^cnn1/bias/Adam/Assign^cnn1/bias/Adam_1/Assign^cnn2/kernel/Adam/Assign^cnn2/kernel/Adam_1/Assign^cnn2/bias/Adam/Assign^cnn2/bias/Adam_1/Assign^cnn3/kernel/Adam/Assign^cnn3/kernel/Adam_1/Assign^cnn3/bias/Adam/Assign^cnn3/bias/Adam_1/Assign
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
¦
save/SaveV2/tensor_namesConst*Ł
valueĻBĢBVariableBbeta1_powerBbeta2_powerB	cnn1/biasBcnn1/bias/AdamBcnn1/bias/Adam_1Bcnn1/kernelBcnn1/kernel/AdamBcnn1/kernel/Adam_1B	cnn2/biasBcnn2/bias/AdamBcnn2/bias/Adam_1Bcnn2/kernelBcnn2/kernel/AdamBcnn2/kernel/Adam_1B	cnn3/biasBcnn3/bias/AdamBcnn3/bias/Adam_1Bcnn3/kernelBcnn3/kernel/AdamBcnn3/kernel/Adam_1*
dtype0*
_output_shapes
:

save/SaveV2/shape_and_slicesConst*=
value4B2B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
Ā
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesVariablebeta1_powerbeta2_power	cnn1/biascnn1/bias/Adamcnn1/bias/Adam_1cnn1/kernelcnn1/kernel/Adamcnn1/kernel/Adam_1	cnn2/biascnn2/bias/Adamcnn2/bias/Adam_1cnn2/kernelcnn2/kernel/Adamcnn2/kernel/Adam_1	cnn3/biascnn3/bias/Adamcnn3/bias/Adam_1cnn3/kernelcnn3/kernel/Adamcnn3/kernel/Adam_1*#
dtypes
2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
l
save/RestoreV2/tensor_namesConst*
valueBBVariable*
dtype0*
_output_shapes
:
h
save/RestoreV2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:

save/AssignAssignVariablesave/RestoreV2*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable*
_output_shapes
: 
q
save/RestoreV2_1/tensor_namesConst* 
valueBBbeta1_power*
dtype0*
_output_shapes
:
j
!save/RestoreV2_1/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:

save/Assign_1Assignbeta1_powersave/RestoreV2_1*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn1/bias*
_output_shapes
: 
q
save/RestoreV2_2/tensor_namesConst* 
valueBBbeta2_power*
dtype0*
_output_shapes
:
j
!save/RestoreV2_2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:

save/Assign_2Assignbeta2_powersave/RestoreV2_2*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn1/bias*
_output_shapes
: 
o
save/RestoreV2_3/tensor_namesConst*
valueBB	cnn1/bias*
dtype0*
_output_shapes
:
j
!save/RestoreV2_3/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
 
save/Assign_3Assign	cnn1/biassave/RestoreV2_3*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn1/bias*
_output_shapes
:
t
save/RestoreV2_4/tensor_namesConst*#
valueBBcnn1/bias/Adam*
dtype0*
_output_shapes
:
j
!save/RestoreV2_4/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
„
save/Assign_4Assigncnn1/bias/Adamsave/RestoreV2_4*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn1/bias*
_output_shapes
:
v
save/RestoreV2_5/tensor_namesConst*%
valueBBcnn1/bias/Adam_1*
dtype0*
_output_shapes
:
j
!save/RestoreV2_5/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
§
save/Assign_5Assigncnn1/bias/Adam_1save/RestoreV2_5*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn1/bias*
_output_shapes
:
q
save/RestoreV2_6/tensor_namesConst* 
valueBBcnn1/kernel*
dtype0*
_output_shapes
:
j
!save/RestoreV2_6/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
°
save/Assign_6Assigncnn1/kernelsave/RestoreV2_6*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn1/kernel*&
_output_shapes
:
v
save/RestoreV2_7/tensor_namesConst*%
valueBBcnn1/kernel/Adam*
dtype0*
_output_shapes
:
j
!save/RestoreV2_7/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
µ
save/Assign_7Assigncnn1/kernel/Adamsave/RestoreV2_7*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn1/kernel*&
_output_shapes
:
x
save/RestoreV2_8/tensor_namesConst*'
valueBBcnn1/kernel/Adam_1*
dtype0*
_output_shapes
:
j
!save/RestoreV2_8/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
·
save/Assign_8Assigncnn1/kernel/Adam_1save/RestoreV2_8*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn1/kernel*&
_output_shapes
:
o
save/RestoreV2_9/tensor_namesConst*
valueBB	cnn2/bias*
dtype0*
_output_shapes
:
j
!save/RestoreV2_9/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
 
save/Assign_9Assign	cnn2/biassave/RestoreV2_9*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn2/bias*
_output_shapes
:
u
save/RestoreV2_10/tensor_namesConst*#
valueBBcnn2/bias/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_10/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
dtypes
2*
_output_shapes
:
§
save/Assign_10Assigncnn2/bias/Adamsave/RestoreV2_10*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn2/bias*
_output_shapes
:
w
save/RestoreV2_11/tensor_namesConst*%
valueBBcnn2/bias/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_11/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_11	RestoreV2
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
dtypes
2*
_output_shapes
:
©
save/Assign_11Assigncnn2/bias/Adam_1save/RestoreV2_11*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn2/bias*
_output_shapes
:
r
save/RestoreV2_12/tensor_namesConst* 
valueBBcnn2/kernel*
dtype0*
_output_shapes
:
k
"save/RestoreV2_12/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_12	RestoreV2
save/Constsave/RestoreV2_12/tensor_names"save/RestoreV2_12/shape_and_slices*
dtypes
2*
_output_shapes
:
²
save/Assign_12Assigncnn2/kernelsave/RestoreV2_12*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn2/kernel*&
_output_shapes
:0
w
save/RestoreV2_13/tensor_namesConst*%
valueBBcnn2/kernel/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_13/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_13	RestoreV2
save/Constsave/RestoreV2_13/tensor_names"save/RestoreV2_13/shape_and_slices*
dtypes
2*
_output_shapes
:
·
save/Assign_13Assigncnn2/kernel/Adamsave/RestoreV2_13*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn2/kernel*&
_output_shapes
:0
y
save/RestoreV2_14/tensor_namesConst*'
valueBBcnn2/kernel/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_14/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_14	RestoreV2
save/Constsave/RestoreV2_14/tensor_names"save/RestoreV2_14/shape_and_slices*
dtypes
2*
_output_shapes
:
¹
save/Assign_14Assigncnn2/kernel/Adam_1save/RestoreV2_14*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn2/kernel*&
_output_shapes
:0
p
save/RestoreV2_15/tensor_namesConst*
valueBB	cnn3/bias*
dtype0*
_output_shapes
:
k
"save/RestoreV2_15/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_15	RestoreV2
save/Constsave/RestoreV2_15/tensor_names"save/RestoreV2_15/shape_and_slices*
dtypes
2*
_output_shapes
:
¢
save/Assign_15Assign	cnn3/biassave/RestoreV2_15*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn3/bias*
_output_shapes
:
u
save/RestoreV2_16/tensor_namesConst*#
valueBBcnn3/bias/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_16/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_16	RestoreV2
save/Constsave/RestoreV2_16/tensor_names"save/RestoreV2_16/shape_and_slices*
dtypes
2*
_output_shapes
:
§
save/Assign_16Assigncnn3/bias/Adamsave/RestoreV2_16*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn3/bias*
_output_shapes
:
w
save/RestoreV2_17/tensor_namesConst*%
valueBBcnn3/bias/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_17/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_17	RestoreV2
save/Constsave/RestoreV2_17/tensor_names"save/RestoreV2_17/shape_and_slices*
dtypes
2*
_output_shapes
:
©
save/Assign_17Assigncnn3/bias/Adam_1save/RestoreV2_17*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn3/bias*
_output_shapes
:
r
save/RestoreV2_18/tensor_namesConst* 
valueBBcnn3/kernel*
dtype0*
_output_shapes
:
k
"save/RestoreV2_18/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_18	RestoreV2
save/Constsave/RestoreV2_18/tensor_names"save/RestoreV2_18/shape_and_slices*
dtypes
2*
_output_shapes
:
²
save/Assign_18Assigncnn3/kernelsave/RestoreV2_18*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn3/kernel*&
_output_shapes
:
w
save/RestoreV2_19/tensor_namesConst*%
valueBBcnn3/kernel/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_19/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_19	RestoreV2
save/Constsave/RestoreV2_19/tensor_names"save/RestoreV2_19/shape_and_slices*
dtypes
2*
_output_shapes
:
·
save/Assign_19Assigncnn3/kernel/Adamsave/RestoreV2_19*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn3/kernel*&
_output_shapes
:
y
save/RestoreV2_20/tensor_namesConst*'
valueBBcnn3/kernel/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_20/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_20	RestoreV2
save/Constsave/RestoreV2_20/tensor_names"save/RestoreV2_20/shape_and_slices*
dtypes
2*
_output_shapes
:
¹
save/Assign_20Assigncnn3/kernel/Adam_1save/RestoreV2_20*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn3/kernel*&
_output_shapes
:
ń
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_20
ś
init_1NoOp^cnn1/kernel/Assign^cnn1/bias/Assign^cnn2/kernel/Assign^cnn2/bias/Assign^cnn3/kernel/Assign^cnn3/bias/Assign^Variable/Assign^beta1_power/Assign^beta2_power/Assign^cnn1/kernel/Adam/Assign^cnn1/kernel/Adam_1/Assign^cnn1/bias/Adam/Assign^cnn1/bias/Adam_1/Assign^cnn2/kernel/Adam/Assign^cnn2/kernel/Adam_1/Assign^cnn2/bias/Adam/Assign^cnn2/bias/Adam_1/Assign^cnn3/kernel/Adam/Assign^cnn3/kernel/Adam_1/Assign^cnn3/bias/Adam/Assign^cnn3/bias/Adam_1/Assign
R
save_1/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_1/StringJoin/inputs_1Const*<
value3B1 B+_temp_0370f8036db74cbfadedfacbae2f2259/part*
dtype0*
_output_shapes
: 
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_1/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_1/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards*
_output_shapes
: 
Ø
save_1/SaveV2/tensor_namesConst*Ł
valueĻBĢBVariableBbeta1_powerBbeta2_powerB	cnn1/biasBcnn1/bias/AdamBcnn1/bias/Adam_1Bcnn1/kernelBcnn1/kernel/AdamBcnn1/kernel/Adam_1B	cnn2/biasBcnn2/bias/AdamBcnn2/bias/Adam_1Bcnn2/kernelBcnn2/kernel/AdamBcnn2/kernel/Adam_1B	cnn3/biasBcnn3/bias/AdamBcnn3/bias/Adam_1Bcnn3/kernelBcnn3/kernel/AdamBcnn3/kernel/Adam_1*
dtype0*
_output_shapes
:

save_1/SaveV2/shape_and_slicesConst*=
value4B2B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
Ō
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesVariablebeta1_powerbeta2_power	cnn1/biascnn1/bias/Adamcnn1/bias/Adam_1cnn1/kernelcnn1/kernel/Adamcnn1/kernel/Adam_1	cnn2/biascnn2/bias/Adamcnn2/bias/Adam_1cnn2/kernelcnn2/kernel/Adamcnn2/kernel/Adam_1	cnn3/biascnn3/bias/Adamcnn3/bias/Adam_1cnn3/kernelcnn3/kernel/Adamcnn3/kernel/Adam_1*#
dtypes
2

save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2*
T0*)
_class
loc:@save_1/ShardedFilename*
_output_shapes
: 
£
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency*
N*
T0*

axis *
_output_shapes
:

save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const*
delete_old_dirs(

save_1/IdentityIdentitysave_1/Const^save_1/control_dependency^save_1/MergeV2Checkpoints*
T0*
_output_shapes
: 
n
save_1/RestoreV2/tensor_namesConst*
valueBBVariable*
dtype0*
_output_shapes
:
j
!save_1/RestoreV2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:

save_1/AssignAssignVariablesave_1/RestoreV2*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable*
_output_shapes
: 
s
save_1/RestoreV2_1/tensor_namesConst* 
valueBBbeta1_power*
dtype0*
_output_shapes
:
l
#save_1/RestoreV2_1/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_1/RestoreV2_1	RestoreV2save_1/Constsave_1/RestoreV2_1/tensor_names#save_1/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
¢
save_1/Assign_1Assignbeta1_powersave_1/RestoreV2_1*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn1/bias*
_output_shapes
: 
s
save_1/RestoreV2_2/tensor_namesConst* 
valueBBbeta2_power*
dtype0*
_output_shapes
:
l
#save_1/RestoreV2_2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_1/RestoreV2_2	RestoreV2save_1/Constsave_1/RestoreV2_2/tensor_names#save_1/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
¢
save_1/Assign_2Assignbeta2_powersave_1/RestoreV2_2*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn1/bias*
_output_shapes
: 
q
save_1/RestoreV2_3/tensor_namesConst*
valueBB	cnn1/bias*
dtype0*
_output_shapes
:
l
#save_1/RestoreV2_3/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_1/RestoreV2_3	RestoreV2save_1/Constsave_1/RestoreV2_3/tensor_names#save_1/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
¤
save_1/Assign_3Assign	cnn1/biassave_1/RestoreV2_3*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn1/bias*
_output_shapes
:
v
save_1/RestoreV2_4/tensor_namesConst*#
valueBBcnn1/bias/Adam*
dtype0*
_output_shapes
:
l
#save_1/RestoreV2_4/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_1/RestoreV2_4	RestoreV2save_1/Constsave_1/RestoreV2_4/tensor_names#save_1/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
©
save_1/Assign_4Assigncnn1/bias/Adamsave_1/RestoreV2_4*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn1/bias*
_output_shapes
:
x
save_1/RestoreV2_5/tensor_namesConst*%
valueBBcnn1/bias/Adam_1*
dtype0*
_output_shapes
:
l
#save_1/RestoreV2_5/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_1/RestoreV2_5	RestoreV2save_1/Constsave_1/RestoreV2_5/tensor_names#save_1/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
«
save_1/Assign_5Assigncnn1/bias/Adam_1save_1/RestoreV2_5*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn1/bias*
_output_shapes
:
s
save_1/RestoreV2_6/tensor_namesConst* 
valueBBcnn1/kernel*
dtype0*
_output_shapes
:
l
#save_1/RestoreV2_6/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_1/RestoreV2_6	RestoreV2save_1/Constsave_1/RestoreV2_6/tensor_names#save_1/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
“
save_1/Assign_6Assigncnn1/kernelsave_1/RestoreV2_6*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn1/kernel*&
_output_shapes
:
x
save_1/RestoreV2_7/tensor_namesConst*%
valueBBcnn1/kernel/Adam*
dtype0*
_output_shapes
:
l
#save_1/RestoreV2_7/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_1/RestoreV2_7	RestoreV2save_1/Constsave_1/RestoreV2_7/tensor_names#save_1/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
¹
save_1/Assign_7Assigncnn1/kernel/Adamsave_1/RestoreV2_7*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn1/kernel*&
_output_shapes
:
z
save_1/RestoreV2_8/tensor_namesConst*'
valueBBcnn1/kernel/Adam_1*
dtype0*
_output_shapes
:
l
#save_1/RestoreV2_8/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_1/RestoreV2_8	RestoreV2save_1/Constsave_1/RestoreV2_8/tensor_names#save_1/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
»
save_1/Assign_8Assigncnn1/kernel/Adam_1save_1/RestoreV2_8*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn1/kernel*&
_output_shapes
:
q
save_1/RestoreV2_9/tensor_namesConst*
valueBB	cnn2/bias*
dtype0*
_output_shapes
:
l
#save_1/RestoreV2_9/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_1/RestoreV2_9	RestoreV2save_1/Constsave_1/RestoreV2_9/tensor_names#save_1/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
¤
save_1/Assign_9Assign	cnn2/biassave_1/RestoreV2_9*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn2/bias*
_output_shapes
:
w
 save_1/RestoreV2_10/tensor_namesConst*#
valueBBcnn2/bias/Adam*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_10/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_10	RestoreV2save_1/Const save_1/RestoreV2_10/tensor_names$save_1/RestoreV2_10/shape_and_slices*
dtypes
2*
_output_shapes
:
«
save_1/Assign_10Assigncnn2/bias/Adamsave_1/RestoreV2_10*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn2/bias*
_output_shapes
:
y
 save_1/RestoreV2_11/tensor_namesConst*%
valueBBcnn2/bias/Adam_1*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_11/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_11	RestoreV2save_1/Const save_1/RestoreV2_11/tensor_names$save_1/RestoreV2_11/shape_and_slices*
dtypes
2*
_output_shapes
:
­
save_1/Assign_11Assigncnn2/bias/Adam_1save_1/RestoreV2_11*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn2/bias*
_output_shapes
:
t
 save_1/RestoreV2_12/tensor_namesConst* 
valueBBcnn2/kernel*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_12/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_12	RestoreV2save_1/Const save_1/RestoreV2_12/tensor_names$save_1/RestoreV2_12/shape_and_slices*
dtypes
2*
_output_shapes
:
¶
save_1/Assign_12Assigncnn2/kernelsave_1/RestoreV2_12*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn2/kernel*&
_output_shapes
:0
y
 save_1/RestoreV2_13/tensor_namesConst*%
valueBBcnn2/kernel/Adam*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_13/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_13	RestoreV2save_1/Const save_1/RestoreV2_13/tensor_names$save_1/RestoreV2_13/shape_and_slices*
dtypes
2*
_output_shapes
:
»
save_1/Assign_13Assigncnn2/kernel/Adamsave_1/RestoreV2_13*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn2/kernel*&
_output_shapes
:0
{
 save_1/RestoreV2_14/tensor_namesConst*'
valueBBcnn2/kernel/Adam_1*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_14/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_14	RestoreV2save_1/Const save_1/RestoreV2_14/tensor_names$save_1/RestoreV2_14/shape_and_slices*
dtypes
2*
_output_shapes
:
½
save_1/Assign_14Assigncnn2/kernel/Adam_1save_1/RestoreV2_14*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn2/kernel*&
_output_shapes
:0
r
 save_1/RestoreV2_15/tensor_namesConst*
valueBB	cnn3/bias*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_15/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_15	RestoreV2save_1/Const save_1/RestoreV2_15/tensor_names$save_1/RestoreV2_15/shape_and_slices*
dtypes
2*
_output_shapes
:
¦
save_1/Assign_15Assign	cnn3/biassave_1/RestoreV2_15*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn3/bias*
_output_shapes
:
w
 save_1/RestoreV2_16/tensor_namesConst*#
valueBBcnn3/bias/Adam*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_16/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_16	RestoreV2save_1/Const save_1/RestoreV2_16/tensor_names$save_1/RestoreV2_16/shape_and_slices*
dtypes
2*
_output_shapes
:
«
save_1/Assign_16Assigncnn3/bias/Adamsave_1/RestoreV2_16*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn3/bias*
_output_shapes
:
y
 save_1/RestoreV2_17/tensor_namesConst*%
valueBBcnn3/bias/Adam_1*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_17/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_17	RestoreV2save_1/Const save_1/RestoreV2_17/tensor_names$save_1/RestoreV2_17/shape_and_slices*
dtypes
2*
_output_shapes
:
­
save_1/Assign_17Assigncnn3/bias/Adam_1save_1/RestoreV2_17*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn3/bias*
_output_shapes
:
t
 save_1/RestoreV2_18/tensor_namesConst* 
valueBBcnn3/kernel*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_18/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_18	RestoreV2save_1/Const save_1/RestoreV2_18/tensor_names$save_1/RestoreV2_18/shape_and_slices*
dtypes
2*
_output_shapes
:
¶
save_1/Assign_18Assigncnn3/kernelsave_1/RestoreV2_18*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn3/kernel*&
_output_shapes
:
y
 save_1/RestoreV2_19/tensor_namesConst*%
valueBBcnn3/kernel/Adam*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_19/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_19	RestoreV2save_1/Const save_1/RestoreV2_19/tensor_names$save_1/RestoreV2_19/shape_and_slices*
dtypes
2*
_output_shapes
:
»
save_1/Assign_19Assigncnn3/kernel/Adamsave_1/RestoreV2_19*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn3/kernel*&
_output_shapes
:
{
 save_1/RestoreV2_20/tensor_namesConst*'
valueBBcnn3/kernel/Adam_1*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_20/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_20	RestoreV2save_1/Const save_1/RestoreV2_20/tensor_names$save_1/RestoreV2_20/shape_and_slices*
dtypes
2*
_output_shapes
:
½
save_1/Assign_20Assigncnn3/kernel/Adam_1save_1/RestoreV2_20*
T0*
validate_shape(*
use_locking(*
_class
loc:@cnn3/kernel*&
_output_shapes
:

save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_2^save_1/Assign_3^save_1/Assign_4^save_1/Assign_5^save_1/Assign_6^save_1/Assign_7^save_1/Assign_8^save_1/Assign_9^save_1/Assign_10^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_16^save_1/Assign_17^save_1/Assign_18^save_1/Assign_19^save_1/Assign_20
1
save_1/restore_allNoOp^save_1/restore_shard"B
save_1/Const:0save_1/Identity:0save_1/restore_all (5 @F8"ŗ
trainable_variables¢
a
cnn1/kernel:0cnn1/kernel/Assigncnn1/kernel/read:02(cnn1/kernel/Initializer/random_uniform:0
P
cnn1/bias:0cnn1/bias/Assigncnn1/bias/read:02cnn1/bias/Initializer/zeros:0
a
cnn2/kernel:0cnn2/kernel/Assigncnn2/kernel/read:02(cnn2/kernel/Initializer/random_uniform:0
P
cnn2/bias:0cnn2/bias/Assigncnn2/bias/read:02cnn2/bias/Initializer/zeros:0
a
cnn3/kernel:0cnn3/kernel/Assigncnn3/kernel/read:02(cnn3/kernel/Initializer/random_uniform:0
P
cnn3/bias:0cnn3/bias/Assigncnn3/bias/read:02cnn3/bias/Initializer/zeros:0"Ī
	variablesĄ½
a
cnn1/kernel:0cnn1/kernel/Assigncnn1/kernel/read:02(cnn1/kernel/Initializer/random_uniform:0
P
cnn1/bias:0cnn1/bias/Assigncnn1/bias/read:02cnn1/bias/Initializer/zeros:0
a
cnn2/kernel:0cnn2/kernel/Assigncnn2/kernel/read:02(cnn2/kernel/Initializer/random_uniform:0
P
cnn2/bias:0cnn2/bias/Assigncnn2/bias/read:02cnn2/bias/Initializer/zeros:0
a
cnn3/kernel:0cnn3/kernel/Assigncnn3/kernel/read:02(cnn3/kernel/Initializer/random_uniform:0
P
cnn3/bias:0cnn3/bias/Assigncnn3/bias/read:02cnn3/bias/Initializer/zeros:0
H

Variable:0Variable/AssignVariable/read:02Variable/initial_value:0
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
l
cnn1/kernel/Adam:0cnn1/kernel/Adam/Assigncnn1/kernel/Adam/read:02$cnn1/kernel/Adam/Initializer/zeros:0
t
cnn1/kernel/Adam_1:0cnn1/kernel/Adam_1/Assigncnn1/kernel/Adam_1/read:02&cnn1/kernel/Adam_1/Initializer/zeros:0
d
cnn1/bias/Adam:0cnn1/bias/Adam/Assigncnn1/bias/Adam/read:02"cnn1/bias/Adam/Initializer/zeros:0
l
cnn1/bias/Adam_1:0cnn1/bias/Adam_1/Assigncnn1/bias/Adam_1/read:02$cnn1/bias/Adam_1/Initializer/zeros:0
l
cnn2/kernel/Adam:0cnn2/kernel/Adam/Assigncnn2/kernel/Adam/read:02$cnn2/kernel/Adam/Initializer/zeros:0
t
cnn2/kernel/Adam_1:0cnn2/kernel/Adam_1/Assigncnn2/kernel/Adam_1/read:02&cnn2/kernel/Adam_1/Initializer/zeros:0
d
cnn2/bias/Adam:0cnn2/bias/Adam/Assigncnn2/bias/Adam/read:02"cnn2/bias/Adam/Initializer/zeros:0
l
cnn2/bias/Adam_1:0cnn2/bias/Adam_1/Assigncnn2/bias/Adam_1/read:02$cnn2/bias/Adam_1/Initializer/zeros:0
l
cnn3/kernel/Adam:0cnn3/kernel/Adam/Assigncnn3/kernel/Adam/read:02$cnn3/kernel/Adam/Initializer/zeros:0
t
cnn3/kernel/Adam_1:0cnn3/kernel/Adam_1/Assigncnn3/kernel/Adam_1/read:02&cnn3/kernel/Adam_1/Initializer/zeros:0
d
cnn3/bias/Adam:0cnn3/bias/Adam/Assigncnn3/bias/Adam/read:02"cnn3/bias/Adam/Initializer/zeros:0
l
cnn3/bias/Adam_1:0cnn3/bias/Adam_1/Assigncnn3/bias/Adam_1/read:02$cnn3/bias/Adam_1/Initializer/zeros:0"
regularization_losses
~
(cnn1/kernel/Regularizer/l2_regularizer:0
(cnn2/kernel/Regularizer/l2_regularizer:0
(cnn3/kernel/Regularizer/l2_regularizer:0"©
	summaries

portfolio:0
cnn1/kernel_1:0
cnn1/bias_1:0
cnn2/kernel_1:0
cnn2/bias_1:0
cnn3/kernel_1:0
cnn3/bias_1:0
mu:0
base_loss:0
full_loss_1:0
sum_p:0"
train_op

Adam