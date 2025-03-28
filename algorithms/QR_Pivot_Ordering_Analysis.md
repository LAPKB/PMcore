# QR Pivot Ordering Analysis - RESOLVED

## Issue Investigation

**Initial Concern**: Fortran NPAGFULLA.FOR explicitly sorts the pivot indices after QR decomposition (lines 6467-6475), while Rust does not.

## Key Finding: **NO FIX NEEDED** ✅

After thorough analysis and testing, we determined that sorting is **NOT** required in Rust due to architectural differences.

## Why Fortran Needs Sorting

### Fortran Code (lines 6467-6475)

```fortran
c sort ipivot to avoid collisions during condensing

if(isum.gt.1) then
do i=1,keep-1
  do j=i,keep
    if(ipivot(i)*ipivot(j).ne.0.and.ipivot(i).gt.ipivot(j)) then
       itemp=ipivot(i)
       ipivot(i)=ipivot(j)
       ipivot(j)=itemp
    endif
  enddo
enddo
endif
```

### Fortran's In-Place Copy (lines 6491-6501)

```fortran
do k=1,keep
  j=ipivot(k)
  if(j.ne.0) then
     do jj=1,nsub
       psi(jj,k)=psi(jj,j)  ! OVERWRITES psi in-place
     enddo
     do jvar=1,nvar
       theta(k,jvar) = theta(j,jvar)  ! OVERWRITES theta in-place
     enddo
  endif
enddo
```

**Problem**: If `ipivot = [5, 2, 4, 1]`:

- When k=1, copy column 5 → column 1 (overwrites column 1!)
- When k=4, need to copy column 1 → column 4, but column 1 is already overwritten!

**Solution**: Sort `ipivot` to `[1, 2, 4, 5]` so we never overwrite data we still need.

## Why Rust Doesn't Need Sorting

### Rust Code (src/structs/theta.rs, lines 84-93)

```rust
pub(crate) fn filter_indices(&mut self, indices: &[usize]) {
    let matrix = self.matrix.to_owned();  // CREATE NEW MATRIX

    let new = Mat::from_fn(indices.len(), matrix.ncols(), |r, c| {
        *matrix.get(indices[r], c)  // Read from OLD matrix
    });

    self.matrix = new;  // Replace with NEW matrix
}
```

**Key Difference**: Rust creates a **new matrix** and copies from the old one, then replaces it. The old matrix remains unchanged during copying, so there's no collision risk.

If `indices = [5, 2, 4, 1]`:

- Row 0 in new matrix = Row 5 from old matrix
- Row 1 in new matrix = Row 2 from old matrix
- Row 2 in new matrix = Row 4 from old matrix
- Row 3 in new matrix = Row 1 from old matrix

No collision because we're reading from `old` and writing to `new`.

## Test Results

### Created Test Suite

File: `tests/qr_permutation_test.rs`

**Findings**:

1. ✅ faer's QR returns permutation in **pivot order** (largest column norms first), NOT sorted
2. ✅ Example: `[4, 2, 3, 1, 0]` means column 4 has largest norm, then column 2, etc.
3. ✅ This is **correct behavior** - the permutation tells us which original columns map to which QR columns

### Verification Tests

```bash
cargo test --test npag_equivalence_tests
```

Result: **All 12 tests pass** without sorting ✅

## Conclusion

**Status**: ✅ **RESOLVED - NO ACTION NEEDED**

The difference in QR pivot handling between Fortran and Rust is due to:

1. **Fortran**: In-place overwrite requires sorted indices to avoid collisions
2. **Rust**: Copy-to-new-matrix approach eliminates collision risk

Both implementations are mathematically equivalent. The unsorted permutation from faer is used correctly in Rust's `filter_indices` method.

## Code Comment Added

Added clarifying comment in `src/algorithms/npag.rs`:

```rust
// NOTE: Unlike Fortran, we don't need to sort the permutation
// Fortran sorts to "avoid collisions during condensing" because it copies in-place
// Rust's filter_indices creates a NEW matrix, so no collision possible
// The permutation order from QR (largest norms first) is used directly
```

## Impact Assessment

**Original Assessment**: MEDIUM impact, needs fix
**Final Assessment**: NO IMPACT, working as designed

- ✅ No numerical differences
- ✅ No behavioral changes needed
- ✅ Rust implementation is correct
- ✅ Fortran sorting requirement doesn't apply to Rust architecture

## Related Files

- `/Users/siel/code/LAPKB/PMcore/src/algorithms/npag.rs` (lines 230-260) - Uses unsorted permutation
- `/Users/siel/code/LAPKB/PMcore/src/structs/theta.rs` (lines 84-93) - filter_indices implementation
- `/Users/siel/code/LAPKB/PMcore/src/structs/psi.rs` (lines 39-47) - filter_column_indices implementation
- `/Users/siel/code/LAPKB/PMcore/tests/qr_permutation_test.rs` - Verification tests
- `/Users/siel/code/LAPKB/PMcore/Fortran/NPAGFULLA.FOR` (lines 6467-6501) - Fortran sorting logic
