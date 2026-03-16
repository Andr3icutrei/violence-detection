import { ComponentFixture, TestBed } from '@angular/core/testing';

import { PortalPage } from './portal-page';

describe('PortalPage', () => {
  let component: PortalPage;
  let fixture: ComponentFixture<PortalPage>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [PortalPage],
    }).compileComponents();

    fixture = TestBed.createComponent(PortalPage);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
