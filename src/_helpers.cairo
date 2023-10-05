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

// fn axis_to_squeeze(mut shape: Span<usize>, axis: usize) -> Span<usize> {

//     // shape = 1 2 1 2 1 = len_from_shape = 4, tensor.data.len() = 4

//     loop {
//         match shape.pop_front() {
//             Option::Some(s) => {
//                 if (index) != *axis {
//                     shape_loop.append(*s);
//                 }
//                 else {
//                     axis_squeezed += 1
//                 }
//             },
//             Option::None(_) => {
//                 break shape_array.span();
//             },
//         };
//     }

//     axis_to_squeeze(shape_array, )

//     // 'shape_array.len()'.print();
//     // shape_array.len().print();
//     //return shape_array.span();

// }

fn squeeze(self: @Tensor<FP16x16>, axes: Option<Span<usize>>) -> Tensor<FP16x16> {
    let mut default_shape: Array<usize> = ArrayTrait::new();
    let shape_loop: Array<usize> = ArrayTrait::new();
    let mut index = 0_usize;

    // 'self.shape.len()'.print();
    // (*self.shape).len().print();    

    // shape = 1 2 1 2 1 = len_from_shape = 4, tensor.data.len() = 4

    let target_axes = match axes {
        Option::Some(mut axes) => {
            let mut axis_squeezed = 0_usize;
            let mut shape = *self.shape;
            loop {
                match axes.pop_front() {
                    Option::Some(axis) => {
                        let mut shape_loop: Array<usize> = ArrayTrait::new();
                        index = 0_usize;
                        loop {
                            match shape.pop_front() {
                                Option::Some(s) => {
                                    // 'shape some'.print();
                                    // (*axis).print();
                                    // index.print();
                                    // axis_squeezed.print();

                                    let squeezed = if *axis >= axis_squeezed {
                                        *axis - axis_squeezed
                                    } else {
                                        *axis
                                    };
                                    // index.print();
                                    // indexito.print();
                                    if index == squeezed {
                                        // 'squeezed'.print();
                                        assert(*s == 1, 'Shape entry not equal to one');
                                        axis_squeezed += 1;
                                    } else {
                                        // 'append'.print();
                                        shape_loop.append(*s);
                                    }
                                },
                                Option::None(_) => {
                                    break;
                                },
                            };
                            index += 1;
                        };
                        shape = shape_loop.span();
                    },
                    Option::None(_) => {
                        break shape;
                    },
                };
            }
        },
        Option::None(_) => {
            let mut default_shape: Array<usize> = ArrayTrait::new();
            let mut input_shape = *self.shape;
            loop {
                match input_shape.pop_front() {
                    Option::Some(input_item) => {
                        if *input_item != 1 {
                            default_shape.append(*input_item);
                        }
                    },
                    Option::None(_) => {
                        break default_shape.span();
                    },
                };
            }
        },
    };

    // 'len_from_shape'.print();
    // len_from_shape(@target_axes).print();
    // '(*self.data).len()'.print();
    // (*self.data).len().print();    

    // let squeezed = TensorTrait::<FP16x16>::new(shape: target_axes, data: *self.data);
    // return squeezed;    
    // return Tensor::<FP16x16> { shape: target_axes, data: *self.data };

    return TensorTrait::<FP16x16>::new(shape: target_axes, data: *self.data);
}
