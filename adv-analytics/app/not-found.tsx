import Link from 'next/link';

export default function NotFound() {
  return (
    <div className="container" style={{ paddingTop: '5rem', textAlign: 'center' }}>
      <div style={{ fontSize: '10rem', fontWeight: 'bold', color: '#4299e1', opacity: '0.5', marginBottom: '1rem' }}>
        404
      </div>
      
      <h1 className="page-title">Page Not Found</h1>
      
      <p className="page-description" style={{ marginBottom: '2rem' }}>
        The page you&apos;re looking for doesn&apos;t exist or has been moved.
      </p>
      
      <div style={{ display: 'flex', justifyContent: 'center', gap: '1rem' }}>
        <Link href="/" className="button button-primary">
          Go to Dashboard
        </Link>
        
        <Link href="/dataset" className="button button-outline">
          View Datasets
        </Link>
      </div>
    </div>
  );
} 