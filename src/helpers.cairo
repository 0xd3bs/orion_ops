use debug::PrintTrait;
use traits::TryInto;
use alexandria_data_structures::array_ext::{SpanTraitExt};
use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{
    Tensor, TensorTrait, FP16x16Tensor, FP16x16TensorAdd, FP16x16TensorMul, FP16x16TensorSub,
    FP16x16TensorDiv
};
use orion::numbers::{FixedTrait, FP16x16, FP16x16Impl};
use orion::numbers::fixed_point::implementations::fp16x16::core::{
    HALF, ONE, FP16x16Mul, FP16x16Div, FP16x16Print, FP16x16IntoI32, FP16x16PartialOrd,
    FP16x16PartialEq
};
use orion::numbers::signed_integer::i32::i32;
fn len_from_shape(shape: @Span<usize>) -> usize {
    let mut result: usize = 1;
    let mut shape = *shape;

    loop {
        match shape.pop_front() {
            Option::Some(item) => {
                result *= *item;
            },
            Option::None(_) => {
                break;
            }
        };
    };

    return result;
}

fn squeeze(self: @Tensor<FP16x16>, axes: Option<Span<usize>>) -> Tensor<FP16x16> {
    let target_shape = match axes {
        Option::Some(mut axes) => {
            let mut axis_squeezed = 0_usize;
            let mut shape = *self.shape;
            'self.shape.len'.print();
            (*self.shape).len().print();
            let shape_len = (*self.shape).len() - 1;
            loop {
                match axes.pop_front() {
                    Option::Some(axis) => {
                        assert(*axis <= shape_len, '*Axis out of accepted range');
                        let mut reshape: Array<usize> = ArrayTrait::new();
                        let mut index = 0_usize;
                        loop {
                            match shape.pop_front() {
                                Option::Some(shape) => {
                                    let squeezed = if *axis >= axis_squeezed {
                                        *axis - axis_squeezed
                                    } else {
                                        *axis
                                    };
                                    if index == squeezed {
                                        assert(*shape == 1, 'Shape entry not equal to one');
                                        axis_squeezed += 1;
                                    } else {
                                        reshape.append(*shape);
                                    }
                                },
                                Option::None(_) => {
                                    break;
                                },
                            };
                            index += 1;
                        };
                        shape = reshape.span();
                    },
                    Option::None(_) => {
                        break shape;
                    },
                };
            }
        },
        Option::None(_) => {
            let mut reshape: Array<usize> = ArrayTrait::new();
            let mut shape = *self.shape;
            loop {
                match shape.pop_front() {
                    Option::Some(shape) => {
                        if *shape != 1 {
                            reshape.append(*shape);
                        }
                    },
                    Option::None(_) => {
                        break reshape.span();
                    },
                };
            }
        },
    };

    return TensorTrait::<FP16x16>::new(shape: target_shape, data: *self.data);
}

fn _squeeze(self: @Tensor<FP16x16>, axes: Option<Span<i32>>) -> Tensor<FP16x16> {
    let target_shape = match axes {
        Option::Some(mut axes) => {
            let mut axis_squeezed = 0;
            let mut shape = *self.shape;
            loop {
                match axes.pop_front() {
                    Option::Some(axis) => {
                        let mut reshape: Array<usize> = ArrayTrait::new();
                        let mut index = 0_usize;
                        let axis = if *axis.sign {
                            assert(*axis.mag <= (*self.shape).len(), 'Axis out of accepted range');
                            (*self.shape).len() - *axis.mag
                        } else {
                            assert(*axis.mag < (*self.shape).len(), 'Axis out of accepted range');
                            *axis.mag
                        };

                        loop {
                            match shape.pop_front() {
                                Option::Some(shape) => {
                                    let squeezed = if axis >= axis_squeezed {
                                        axis - axis_squeezed
                                    } else {
                                        axis
                                    };
                                    if index == squeezed {
                                        assert(*shape == 1, 'Shape entry not equal to one');
                                        axis_squeezed += 1;
                                    } else {
                                        reshape.append(*shape);
                                    }
                                },
                                Option::None(_) => {
                                    break;
                                },
                            };
                            index += 1;
                        };
                        shape = reshape.span();
                    },
                    Option::None(_) => {
                        break shape;
                    },
                };
            }
        },
        Option::None(_) => {
            let mut reshape: Array<usize> = ArrayTrait::new();
            let mut shape = *self.shape;
            loop {
                match shape.pop_front() {
                    Option::Some(shape) => {
                        if *shape != 1 {
                            reshape.append(*shape);
                        }
                    },
                    Option::None(_) => {
                        break reshape.span();
                    },
                };
            }
        },
    };

    return TensorTrait::<FP16x16>::new(shape: target_shape, data: *self.data);
}
