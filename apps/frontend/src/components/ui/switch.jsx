import * as SwitchPrimitive from '@radix-ui/react-switch'
import { cn } from '@/lib/utils'

export function Switch({ className, checked, ...props }) {
  return (
    <SwitchPrimitive.Root
      checked={checked}
      className={cn(
        'peer inline-flex h-5 w-9 shrink-0 cursor-pointer items-center rounded-full transition-colors',
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 focus-visible:ring-offset-background',
        'disabled:cursor-not-allowed disabled:opacity-50',
        className
      )}
      style={{
        backgroundColor: checked ? 'hsl(213 94% 68%)' : 'hsl(220 9% 18%)',
        border: '1px solid hsl(216 12% 24%)',
      }}
      {...props}
    >
      <SwitchPrimitive.Thumb
        className="pointer-events-none block h-3.5 w-3.5 rounded-full bg-white shadow-md ring-0 transition-transform"
        style={{
          transform: checked ? 'translateX(18px)' : 'translateX(2px)',
        }}
      />
    </SwitchPrimitive.Root>
  )
}
