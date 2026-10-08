//! Host-storage access shared by owned arrays and borrowed views.

mod sealed {
    pub trait Sealed {}

    impl<T> Sealed for Vec<T> {}
    impl<T> Sealed for &[T] {}
    impl<T> Sealed for &mut [T] {}
}

/// Read-only access to a contiguous buffer of `T`. Sealed: the only buffers
/// are `Vec<T>` (owned), `&[T]`, and `&mut [T]`.
pub trait Buffer<T>: sealed::Sealed {
    /// Buffer lent by view adapters (`slice`, `t_view`, ...). An immutable view
    /// lends its own `&'a [T]`, so chained adapters outlive temporaries.
    type Reborrow<'s>: Buffer<T>
    where
        Self: 's;

    fn as_slice(&self) -> &[T];
    fn reborrow(&self) -> Self::Reborrow<'_>;
}

/// Mutable access to a contiguous buffer of `T`.
pub trait BufferMut<T>: Buffer<T> {
    fn as_mut_slice(&mut self) -> &mut [T];
}

impl<T> Buffer<T> for Vec<T> {
    type Reborrow<'s>
        = &'s [T]
    where
        T: 's;

    fn as_slice(&self) -> &[T] {
        self
    }
    fn reborrow(&self) -> &[T] {
        self
    }
}

impl<T> BufferMut<T> for Vec<T> {
    fn as_mut_slice(&mut self) -> &mut [T] {
        self
    }
}

impl<'a, T> Buffer<T> for &'a [T] {
    type Reborrow<'s>
        = &'a [T]
    where
        'a: 's;

    fn as_slice(&self) -> &[T] {
        self
    }
    fn reborrow(&self) -> &'a [T] {
        self
    }
}

impl<'a, T> Buffer<T> for &'a mut [T] {
    type Reborrow<'s>
        = &'s [T]
    where
        'a: 's;

    fn as_slice(&self) -> &[T] {
        self
    }
    fn reborrow(&self) -> &[T] {
        self
    }
}

impl<T> BufferMut<T> for &mut [T] {
    fn as_mut_slice(&mut self) -> &mut [T] {
        self
    }
}
