import { ChangeDetectorRef, Component, OnDestroy, OnInit } from '@angular/core';
import { MainLayoutPage } from '../main-layout/main-layout-page.type';
import { NavigationEnd, NavigationExtras, Router } from '@angular/router';
import { TranslatePipe } from '@ngx-translate/core';
import { SidebarService } from '../../services/sidebar/sidebar.service';
import { Subscription } from 'rxjs';
import { MainLayout } from '../main-layout/main-layout';
import { AuthService } from '../../services/auth/auth-service';
import { UserResponseDto } from '../../core/api/models/user-response-dto';

export interface SidebarMenuItem {
  roles: string[];
  id: MainLayoutPage;
  title: string;
  icon: string;
}

export const menuItems: SidebarMenuItem[] = [
  {
    id: 'dashboard',
    roles: ['admin', 'user'],
    title: 'sidebar.dashboard',
    icon: 'fa-solid fa-house',
  },
  {
    id: 'users',
    roles: ['admin'],
    title: 'sidebar.users',
    icon: 'fa-solid fa-users',
  },
  {
    id: 'review-datasets',
    roles: ['admin'],
    title: 'sidebar.review-datasets',
    icon: 'fa-solid fa-layer-group',
  },
  {
    id: 'stats',
    roles: ['admin'],
    title: 'sidebar.stats',
    icon: 'fa-solid fa-chart-line',
  },
];

@Component({
  selector: 'app-sidebar',
  imports: [TranslatePipe],
  templateUrl: './sidebar.html',
  styleUrl: './sidebar.css',
})
export class Sidebar implements OnInit, OnDestroy {
  selectedSidebarItem: MainLayoutPage = 'dashboard';
  sidebarSubscription: Subscription | null = null;
  visibleMenuItemsForUser: SidebarMenuItem[] = [];

  constructor(
    private cdr: ChangeDetectorRef,
    private router: Router,
    private sidebarService: SidebarService,
    private authService: AuthService,
  ) {
    this.sidebarSubscription = this.sidebarService.refresh$.subscribe((page: MainLayoutPage) => {
      this.syncSelectedSidebarItem(page);
    });
  }
  ngOnInit(): void {
    this.authService.me().subscribe({
      next: (data: UserResponseDto) => {
        if (data.is_admin) {
          this.visibleMenuItemsForUser = menuItems;
        } else {
          this.visibleMenuItemsForUser = menuItems.filter(item => item.roles.includes('user'));
        }
        this.cdr.detectChanges();
      },
      error: () => {
        this.router.navigate(['/']);
      }
    })
  }

  public goToPage(sidebarItem: MainLayoutPage): void {
    this.selectedSidebarItem = sidebarItem;

    this.router.navigate(['/' + sidebarItem]);
  }

  private syncSelectedSidebarItem(page: MainLayoutPage): void {
    this.selectedSidebarItem = page;
  }

  ngOnDestroy(): void {
    if (this.sidebarSubscription) {
      this.sidebarSubscription.unsubscribe();
    }
  }
}
