import { cn } from '@/lib/utils'

export function Card({ className, ...props }) {
  return (
    <div
      className={cn('rounded-lg border border-border bg-muted/30 p-4', className)}
      {...props}
    />
  )
}

export function CardHeader({ className, ...props }) {
  return <div className={cn('mb-3 flex items-center justify-between', className)} {...props} />
}

export function CardTitle({ className, ...props }) {
  return <p className={cn('text-xs font-medium uppercase tracking-wider text-muted-foreground', className)} {...props} />
}
