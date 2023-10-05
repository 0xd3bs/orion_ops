use debug::PrintTrait;
use traits::TryInto;
use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{
    Tensor, TensorTrait, FP16x16Tensor, FP16x16TensorAdd, FP16x16TensorMul, FP16x16TensorSub,
    FP16x16TensorDiv
};
use orion::numbers::{FixedTrait, FP16x16, FP16x16Impl};
use orion::numbers::fixed_point::implementations::fp16x16::core::{
    HALF, ONE, FP16x16Mul, FP16x16Div, FP16x16IntoI32, FP16x16PartialOrd, FP16x16PartialEq
};

use orion_ops::{generated::{Squeeze_test::Squeeze_test}};

use orion_ops::{helpers::{squeeze, _squeeze}};

use orion::numbers::signed_integer::i32::i32;

#[test]
#[available_gas(99999999999999999)]
fn test() {
    let mut data_result = ArrayTrait::<FP16x16>::new();

    // shape = 1 2 1 2 1 = len_from_shape = 4, tensor.data.len() = 4

    // let output_squeeze_none = squeeze(@(Squeeze_test()), Option::None(())); // None  
    // let mut shape_none = output_squeeze_none.shape;

    // let output_squeeze_0 = squeeze(@(Squeeze_test()), Option::Some(array![0].span())); // 0
    // let mut shapee_0 = output_squeeze_0.shape;

    // let output_squeeze_0_2 = squeeze(@(Squeeze_test()), Option::Some(array![0, 2].span()));
    // let mut shape_0_2 = output_squeeze_0_2.shape;

    // let output_squeeze_out_range = squeeze(@(Squeeze_test()), Option::Some(array![5].span()));
    // let mut shape_out_range = output_squeeze_out_range.shape;

    // let output_squeeze_negative = _squeeze(@(Squeeze_test()), Option::Some(array![-1].span()));
    // let mut shape_negative = output_squeeze_negative.shape;

    let output_squeeze_negative = _squeeze(
        @(Squeeze_test()),
        Option::Some(
            array![
                i32 { mag: 5, sign: true }, i32 { mag: 3, sign: true }, i32 { mag: 4, sign: false }
            ]
                .span()
        )
    );
    let mut shape_negative = output_squeeze_negative.shape;

    // '(shape_none)'.print();
    // loop {
    //     match shape_none.pop_front() {
    //         Option::Some(item) => {
    //             (*item).print();
    //         },
    //         Option::None(_) => {
    //             break;
    //         }
    //     };
    // };

    // '(shapee_0)'.print();
    // loop {
    //     match shapee_0.pop_front() {
    //         Option::Some(item) => {
    //             (*item).print();
    //         },
    //         Option::None(_) => {
    //             break;
    //         }
    //     };
    // };

    // '(shapee_0)'.print();
    // loop {
    //     match shapee_0.pop_front() {
    //         Option::Some(item) => {
    //             (*item).print();
    //         },
    //         Option::None(_) => {
    //             break;
    //         }
    //     };
    // };

    // '(shape_0_2)'.print();
    // loop {
    //     match shape_0_2.pop_front() {
    //         Option::Some(item) => {
    //             (*item).print();
    //         },
    //         Option::None(_) => {
    //             break;
    //         }
    //     };
    // };

    // '(shape_out_range)'.print();
    // loop {
    //     match shape_out_range.pop_front() {
    //         Option::Some(item) => {
    //             (*item).print();
    //         },
    //         Option::None(_) => {
    //             break;
    //         }
    //     };
    // };

    '(shape_negative)'.print();
    loop {
        match shape_negative.pop_front() {
            Option::Some(item) => {
                (*item).print();
            },
            Option::None(_) => {
                break;
            }
        };
    };

    // 'output_squeeze.0'.print();
    // (*output_squeeze.data.at(0)).print();
    // 'output_squeeze.1'.print();
    // (*output_squeeze.data.at(1)).print();

    assert(1_u32 < 2_u32, 'no');
}
