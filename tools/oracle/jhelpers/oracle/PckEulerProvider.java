package oracle;

import org.hipparchus.CalculusFieldElement;
import org.hipparchus.geometry.euclidean.threed.Rotation;
import org.orekit.frames.FieldTransform;
import org.orekit.frames.Transform;
import org.orekit.frames.TransformProvider;
import org.orekit.time.AbsoluteDate;
import org.orekit.time.FieldAbsoluteDate;
import org.orekit.time.TimeScale;

/** Parent (ICRF-aligned) to body-fixed rotation from a type-2 binary PCK segment (3-1-3 Euler angles
 *  phi, theta, psi; C = R3(psi) R1(theta) R3(phi)). Records are passed in raw, exactly as stored in the
 *  DAF array: [MID, RADIUS, 3 x ncoef Chebyshev coefficients] per record. Rotation only (zero rate):
 *  the gravity model needs the static orientation, nothing else uses this frame. */
public class PckEulerProvider implements TransformProvider {
    private final double init, intlen;
    private final int rsize, ncoef;
    private final double[] records;
    private final AbsoluteDate j2000Tdb;
    private final TimeScale tdb;

    public PckEulerProvider(double init, double intlen, int rsize, double[] records,
                            AbsoluteDate j2000Tdb, TimeScale tdb) {
        this.init = init;
        this.intlen = intlen;
        this.rsize = rsize;
        this.ncoef = (rsize - 2) / 3;
        this.records = records;
        this.j2000Tdb = j2000Tdb;
        this.tdb = tdb;
    }

    public double[][] matrix(double et) {
        int n = records.length / rsize;
        int idx = (int) Math.floor((et - init) / intlen);
        if (idx == n && et == init + n * intlen) {
            idx = n - 1;
        }
        if (idx < 0 || idx >= n) {
            throw new IllegalArgumentException("et " + et + " outside loaded PCK records");
        }
        int off = idx * rsize;
        double x = (et - records[off]) / records[off + 1];
        double[] ang = new double[3];
        for (int c = 0; c < 3; ++c) {
            int base = off + 2 + c * ncoef;
            double t0 = 1.0, t1 = x, sum = records[base] + records[base + 1] * x;
            for (int k = 2; k < ncoef; ++k) {
                double t2 = 2.0 * x * t1 - t0;
                sum += records[base + k] * t2;
                t0 = t1;
                t1 = t2;
            }
            ang[c] = sum;
        }
        double[][] r3psi = r3(ang[2]), r1th = r1(ang[1]), r3phi = r3(ang[0]);
        return mul(r3psi, mul(r1th, r3phi));
    }

    private static double[][] r1(double a) {
        double c = Math.cos(a), s = Math.sin(a);
        return new double[][] {{1, 0, 0}, {0, c, s}, {0, -s, c}};
    }

    private static double[][] r3(double a) {
        double c = Math.cos(a), s = Math.sin(a);
        return new double[][] {{c, s, 0}, {-s, c, 0}, {0, 0, 1}};
    }

    private static double[][] mul(double[][] a, double[][] b) {
        double[][] m = new double[3][3];
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                m[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
            }
        }
        return m;
    }

    @Override
    public Transform getTransform(AbsoluteDate date) {
        double et = date.offsetFrom(j2000Tdb, tdb);
        return new Transform(date, new Rotation(matrix(et), 1.0e-10));
    }

    @Override
    public <T extends CalculusFieldElement<T>> FieldTransform<T> getTransform(FieldAbsoluteDate<T> date) {
        throw new UnsupportedOperationException("field transforms are not used by the oracle harness");
    }
}
